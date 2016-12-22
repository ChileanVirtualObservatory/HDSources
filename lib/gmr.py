import copy
import numba
import numpy as np



################################################################
# HELPER FUNCTIONS
################################################################

@numba.jit('float64[:,:] (float64[:], float64[:])')
def _outer(x, y):
    m = x.shape[0]
    n = y.shape[0]
    res = np.empty((m, n), dtype=np.float64)
    for i in range(m):
        for j in range(n):
            res[i, j] = x[i]*y[j]
    return res

@numba.jit('float64 (float64[:,:])')
def _det2D(X):
    return X[0,0]*X[1,1] - X[0,1]*X[1,0]



#################################################################
# MOMENT PRESERVING GAUSSIAN
#################################################################

@numba.jit('Tuple((float64, float64[:], float64[:,:])) (float64, float64[:], float64[:,:], float64, float64[:], float64[:,:])', nopython=True)
def merge(c1, mu1, sig1, c2, mu2, sig2):
    c_m = c1+c2
    mu_m = (c1/c_m)*mu1 + (c2/c_m)*mu2
    sig_m = (c1/c_m)*sig1 + (c2/c_m)*sig2 + (c1/c_m)*(c2/c_m)*_outer(mu1-mu2, mu1-mu2)
    return (c_m, mu_m, sig_m) 



#################################################################
# KL-DIVERGENCE UPPER BOUND
# ref: A Kullback-Leibler Approach to Gaussian Mixture Reduction
#################################################################

@numba.jit('float64 (float64, float64[:], float64[:,:], float64, float64[:], float64[:,:])', nopython=True)
def KL_dissimilarity(c1, mu1, sig1, c2, mu2, sig2):
    # merged moment preserving gaussian
    c_m, mu_m, sig_m = merge(c1, mu1, sig1, c2, mu2, sig2)
    # KL divergence upper bound as proposed in: A Kullback-Leibler Approach to Gaussian Mixture Reduction
    return 0.5*((c1+c2)*np.log(_det2D(sig_m)) - c1*np.log(_det2D(sig1)) - c2*np.log(_det2D(sig2)))




#################################################################
# MAIN GAUSSIAN REDUCTION FUNCTION
#################################################################

def gaussian_reduction(c, mu, sig, n_comp, metric=KL_dissimilarity):
    c = c.tolist(); mu = map(np.array, mu.tolist()); sig = [(s**2)*np.identity(2) for s in sig]
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
                diss = metric(c[i], mu[i], sig[i], c[j], mu[j], sig[j])
                if diss < diss_min: i_min = i; j_min = j; diss_min = diss
        # compute the moment preserving  merged gaussian
        c_m, mu_m, sig_m = merge(c[i_min], mu[i_min], sig[i_min], c[j_min], mu[j_min], sig[j_min])
        # updating structures
        print('Merged components {0} and {1} with {2} dissimilarity'.format(i_min, j_min, diss_min))
        del c[max(i_min, j_min)]; del c[min(i_min, j_min)]; c.append(c_m)
        del mu[max(i_min, j_min)]; del mu[min(i_min, j_min)]; mu.append(mu_m)
        del sig[max(i_min, j_min)]; del sig[min(i_min, j_min)]; sig.append(sig_m)
        new_component = components.pop(max(i_min,j_min)) + components.pop(min(i_min,j_min))
        new_component.sort()
        components.append(new_component)
        components_dict[m-1] = copy.deepcopy(components)
    return components_dict