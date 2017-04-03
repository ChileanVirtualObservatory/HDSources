import copy
import numba
import numpy as np



################################################################
# HELPER FUNCTIONS
################################################################

@numba.jit('float64[:,:] (float64[:], float64[:])', nopython=True)
def _outer(x, y):
    m = x.shape[0]
    n = y.shape[0]
    res = np.empty((m, n), dtype=np.float64)
    for i in range(m):
        for j in range(n):
            res[i, j] = x[i]*y[j]
    return res

@numba.jit('float64 (float64[:,:])', nopython=True)
def _det2D(X):
    return X[0,0]*X[1,1] - X[0,1]*X[1,0]


@numba.jit('float64 (float64[:,:])', nopython=True)
def _det3D(X):
    return X[0,0] * (X[1,1] * X[2,2] - X[2,1] * X[1,2]) - \
           X[1,0] * (X[0,1] * X[2,2] - X[2,1] * X[0,2]) + \
           X[2,0] * (X[0,1] * X[1,2] - X[1,1] * X[0,2])


def normal(x, mu, sig):
    d = mu.shape[0]
    return (1./np.sqrt((2.*np.pi)**d * np.linalg.det(sig))) * np.exp(-0.5*np.dot(x-mu, np.dot(np.linalg.inv(sig), x-mu)))



#################################################################
# MOMENT PRESERVING GAUSSIAN
#################################################################

@numba.jit('Tuple((float64, float64[:], float64[:,:])) (float64, float64[:], float64[:,:], float64, float64[:], float64[:,:])', nopython=True)
def merge(c1, mu1, sig1, c2, mu2, sig2):
    c_m = c1+c2
    mu_m = (c1/c_m)*mu1 + (c2/c_m)*mu2
    sig_m = (c1/c_m)*sig1 + (c2/c_m)*sig2 + (c1/c_m)*(c2/c_m)*_outer(mu1-mu2, mu1-mu2)
    return (c_m, mu_m, sig_m)


###########################################################################
# ISD: Integral Square Difference
# ref: Cost-Function-Based Gaussian Mixture Reduction for Target Tracking
###########################################################################
def ISD_dissimilarity(w1, mu1, sig1, w2, mu2, sig2):
    # merged moment preserving gaussian
    w_m, mu_m, sig_m = merge(w1, mu1, sig1, w2, mu2, sig2)
    # ISD analytical computation between merged component and the pair of gaussians
    Jhr = w1*w_m * normal(mu1, mu_m, sig1+sig_m) + w2*w_m * normal(mu2, mu_m, sig2+sig_m)
    Jrr = w_m**2 * (1./np.sqrt((2*np.pi)**2 * np.linalg.det(2*sig_m)))
    Jhh = (w1**2)*(1./np.sqrt((2*np.pi)**2 * np.linalg.det(2*sig1))) + \
          (w2**2)*(1./np.sqrt((2*np.pi)**2 * np.linalg.det(2*sig2))) + \
          2*w1*w2*normal(mu1, mu2, sig1+sig2)
    return Jhh - 2*Jhr + Jrr

#normalized version
def ISD_dissimilarity_(w1, mu1, sig1, w2, mu2, sig2):
    _w1 = w1 / (w1 + w2)
    _w2 = w2 / (w1 + w2)
    # merged moment preserving gaussian
    w_m, mu_m, sig_m = merge(_w1, mu1, sig1, _w2, mu2, sig2)
    # ISD analytical computation between merged component and the pair of gaussians
    Jhr = _w1*w_m * normal(mu1, mu_m, sig1+sig_m) + _w2*w_m * normal(mu2, mu_m, sig2+sig_m)
    Jrr = w_m**2 * (1./np.sqrt((2*np.pi)**2 * np.linalg.det(2*sig_m)))
    Jhh = (_w1**2)*(1./np.sqrt((2*np.pi)**2 * np.linalg.det(2*sig1))) + \
          (_w2**2)*(1./np.sqrt((2*np.pi)**2 * np.linalg.det(2*sig2))) + \
          2*_w1*_w2*normal(mu1, mu2, sig1+sig2)
    return Jhh - 2*Jhr + Jrr


#################################################################
# KL-DIVERGENCE UPPER BOUND
# ref: A Kullback-Leibler Approach to Gaussian Mixture Reduction
#################################################################

@numba.jit('float64 (float64, float64[:], float64[:,:], float64, float64[:], float64[:,:])', nopython=True)
def KL_dissimilarity(c1, mu1, sig1, c2, mu2, sig2):
    # merged moment preserving gaussian
    c_m, mu_m, sig_m = merge(c1, mu1, sig1, c2, mu2, sig2)
    # KL divergence upper bound as proposed in: A Kullback-Leibler Approach to Gaussian Mixture Reduction
    if len(sig_m)==2:
        return 0.5*((c1+c2)*np.log(_det2D(sig_m)) - c1*np.log(_det2D(sig1)) - c2*np.log(_det2D(sig2)))
    else:
        return 0.5*((c1+c2)*np.log(_det3D(sig_m)) - c1*np.log(_det3D(sig1)) - c2*np.log(_det3D(sig2)))


#################################################################
# MAIN GAUSSIAN REDUCTION FUNCTION
#################################################################

def gaussian_reduction(c, mu, sig, n_comp, metric='KL', verbose=True):
    if metric=='KL': 
        _metric = KL_dissimilarity
        isd_hist = list(); kl_hist = list()
    elif metric=='ISD': 
        _metric = ISD_dissimilarity
        isd_hist = list(); kl_hist = None
    elif metric=='ISD_':
        _metric = ISD_dissimilarity_
        isd_hist = list(); kl_hist = None
    else: return None

    d = mu.shape[1]
    c = c.tolist()
    mu = list(map(np.array, mu.tolist()))
    if d==2: sig = [(s**2)*np.identity(2) for s in sig]
    elif d==3: sig = [(s**2)*np.identity(3) for s in sig]
    # indexes of the actual gaussian components
    components = [[i] for i in range(len(c))]
    components_dict = {len(components) : copy.deepcopy(components)}

    # main loop
    while len(components)>n_comp:
        m = len(c)
        diss_min = np.inf
        i_min = -1; j_min = -1
        for i in range(m):
            for j in range(i+1,m):
                diss = _metric(c[i], mu[i], sig[i], c[j], mu[j], sig[j])
                if diss < diss_min: i_min = i; j_min = j; diss_min = diss
        # compute the moment preserving  merged gaussian
        c_m, mu_m, sig_m = merge(c[i_min], mu[i_min], sig[i_min], c[j_min], mu[j_min], sig[j_min])
        # updating structures
        if (metric=='ISD' or metric=='ISD_') and verbose:
            print('Merged components {0} and {1} with {2} ISD dist'.format(i_min, j_min, diss_min))
            isd_hist.append(diss_min)    
        elif metric=='KL' and verbose:
            ISD_diss = ISD_dissimilarity(c[i_min], mu[i_min], sig[i_min], c[j_min], mu[j_min], sig[j_min])
            print('Merged components {0} and {1} with {2} KL dist and {3} ISD dist'.format(i_min, j_min, diss_min, ISD_diss))
            isd_hist.append(ISD_diss), kl_hist.append(diss_min)
        del c[max(i_min, j_min)]; del c[min(i_min, j_min)]; c.append(c_m)
        del mu[max(i_min, j_min)]; del mu[min(i_min, j_min)]; mu.append(mu_m)
        del sig[max(i_min, j_min)]; del sig[min(i_min, j_min)]; sig.append(sig_m)
        new_component = components.pop(max(i_min,j_min)) + components.pop(min(i_min,j_min))
        new_component.sort()
        components.append(new_component)
        components_dict[m-1] = copy.deepcopy(components)
    return components_dict, isd_hist, kl_hist
