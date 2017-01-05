import numba
import numpy as np
import numpy.ma as ma
import scipy as sp
import numexpr as ne
from math import sqrt, exp


##################################################################
# GLOBAL VARIABLES
##################################################################
supp = 5.      # gaussians support
minsig = 0.001 # guassians minimal broadening

@numba.jit('float64[:] (float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64, float64)', nopython=True)
def u_eval(c, sig, xe, ye, xc, yc, supp=supp, sig0=minsig):
    m = len(xe)
    n = len(xc)
    ret = np.zeros(m)
    for i in range(m):
        for j in range(n):
            dist2 = (xe[i]-xc[j])**2 + (ye[i]-yc[j])**2
            if  dist2 > supp**2 * sig[j]**2: continue
            ret[i] += c[j] * exp( -dist2 / (2* (sig0**2 + sig[j]**2) ) )
    return ret 


@numba.jit('float64[:,:] (float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64, float64)', nopython=True)
def u_eval_full(c, sig, xe, ye, xc, yc, supp=supp, sig0=minsig):
    m = len(xe)
    n = len(xc)
    ret = np.zeros((6,m))
    for i in range(m):
        for j in range(n):
            dist2 = (xe[i]-xc[j])**2 + (ye[i]-yc[j])**2
            if  dist2 > supp**2 * sig[j]**2: continue
            # common terms
            aux1 = (-1./(sig0**2+sig[j]**2))
            aux2 = aux1**2
            # u evaluation
            ret[0,i] += c[j] * exp( -dist2 / (2* (sig0**2 + sig[j]**2) ) )
            # ux evaluation
            ret[1,i] += aux1 * ret[0,i] * (xe[i]-xc[j])
            # uy evaluation
            ret[2,i] += aux1 * ret[0,i] * (ye[i]-yc[j])
            # uxy evaluation
            ret[3,i] += aux2 * ret[0,i] * (xe[i]-xc[j])*(ye[i]-yc[j])
            # uxx evaluation (REVISAR ESTO)
            ret[4,i] += aux2 * ret[0,i] * ((xe[i]-xc[j])**2 - sig0**2 - sig[j]**2)
            # uyy evaluation (REVISAR ESTO)
            ret[5,i] += aux2 * ret[0,i] * ((ye[i]-yc[j])**2 - sig0**2 - sig[j]**2)
    return ret



###################################################################
# RBF (Gaussian) functions and its derivatives
#
# NOTE: For the compact support implementation, we
# take the fact that gaussian functions decrease faster
# than polynomial functions
###################################################################

def phi(x, y, sig, sig0=minsig, supp=5.):
    retval = ne.evaluate('exp(-(x**2+y**2)/(2*(sig0**2+sig**2)))')
    if supp!=0.: retval[retval < np.exp(-0.5 * supp**2)] = 0.
    return retval

def phix(x, y, sig, sig0=minsig, supp=5.):
    retval = ne.evaluate('(-1./(sig0**2+sig**2)) * exp(-(x**2+y**2)/(2*(sig0**2+sig**2))) * x')
    if supp!=0.: retval[retval < np.exp(-0.5 * supp**2)] = 0.
    return retval

def phiy(x, y, sig, sig0=minsig, supp=5.):
    retval = ne.evaluate('(-1./(sig0**2+sig**2)) * exp(-(x**2+y**2)/(2*(sig0**2+sig**2))) * y')
    if supp!=0.: retval[retval < np.exp(-0.5 * supp**2)] = 0.
    return retval

#same as phiyx
def phixy(x, y, sig, sig0=minsig, supp=5.):
    retval = ne.evaluate('(1./(sig0**2+sig**2)**2) * exp(-(x**2+y**2)/(2*(sig0**2+sig**2))) * (x*y)')
    if supp!=0.: retval[retval < np.exp(-0.5 * supp**2)] = 0.
    return retval

def phixx(x, y, sig, sig0=minsig, supp=5.):
    retval = ne.evaluate('(1./(sig0**2+sig**2)**2) * exp(-(x**2+y**2)/(2*(sig0**2+sig**2))) * (x**2 - sig0**2 - sig**2)')
    if supp!=0.: retval[retval < np.exp(-0.5 * supp**2)] = 0.
    return retval

def phiyy(x, y, sig, sig0=minsig, supp=5.):
    retval = ne.evaluate('(1./(sig0**2+sig**2)**2) * exp(-(x**2+y**2)/(2*(sig0**2+sig**2))) * (y**2 - sig0**2 - sig**2)')
    if supp!=0.: retval[retval < np.exp(-0.5 * supp**2)] = 0.
    return retval


def estimate_rms(data):
    """
    Computes RMS value of an N-dimensional numpy array
    """

    if isinstance(data, ma.MaskedArray):
        ret = np.sum(data*data) / (np.size(data) - np.sum(data.mask)) 
    else: 
        ret = np.sum(data*data) / np.size(data)
    return np.sqrt(ret)


def estimate_entropy(data):
    """
    Computes Entropy of an N-dimensional numpy array
    """

    # estimation of probabilities
    p = np.histogram(data.ravel(), bins=256, density=False)[0].astype(float)
    # little fix for freq=0 cases
    p = (p+1.)/(p.sum()+256.)
    # computation of entropy 
    return -np.sum(p * np.log2(p))


def estimate_variance(data):
    """
    Computes variance of an N-dimensional numpy array
    """

    return np.std(data)**2


def compute_residual_stats(data, xc, yc, c, sig, dims, square_c=True, compact_supp=True):
    """
    Computes the residual stats between appproximation and real data
    """

    _xe = np.linspace(0., 1., dims[0]+2)[1:-1]
    _ye = np.linspace(0., 1., dims[1]+2)[1:-1]
    Xe,Ye = np.meshgrid(_xe, _ye, sparse=False, indexing='ij')
    xe = Xe.ravel(); ye = Ye.ravel()

    if square_c: c = c**2
    u = u_eval(c, sig, xe, ye, xc, yc, supp=supp, sig0=minsig)
    u = u.reshape(dims)
    residual = data-u
    
    return (estimate_variance(residual), 
            estimate_entropy(residual),
            estimate_rms(residual))


def build_dist_matrix(points):
    """
    Builds a distance matrix from points array.
    It returns a (n_points, n_points) distance matrix. 

    points: NumPy array with shape (n_points, 2) 
    """
    xp = points[:,0]
    yp = points[:,1]
    N = points.shape[0]
    Dx = np.empty((N,N))
    Dy = np.empty((N,N))
    for k in range(N):
        Dx[k,:] = xp[k]-xp
        Dy[k,:] = yp[k]-yp
    return np.sqrt(Dx**2+Dy**2)


def load_data(fit_path):
    container = load_fits(fit_path)
    data = standarize(container.primary)[0]
    data = data.data

    # stacking it
    data = data.sum(axis=0)
    data -= data.min()
    data /= data.max()

    # generating the data function
    x = np.linspace(0., 1., data.shape[0]+2, endpoint=True)[1:-1]
    y = np.linspace(0., 1., data.shape[1]+2, endpoint=True)[1:-1]
    _dfunc = RegularGridInterpolator((x,y), data, method='linear', bounds_error=False, fill_value=0.)
    
    def dfunc(points):
        if points.ndim==1:
            return _dfunc([[points[1],points[0]]])
        elif points.ndim==2:
            return  _dfunc(points[:,::-1])
    
    return x, y, data, dfunc