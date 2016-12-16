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
def evaluate(c, sig, xe, ye, xc, yc, supp=5., sig0=0.001):
    m = len(xe)
    n = len(xc)
    ret = np.zeros(m)
    for i in range(m):
        for j in range(n):
            if (xe[i]-xc[j])**2 + (ye[i]-yc[j])**2 > supp**2 * sig[j]**2: continue
            ret[i] += c[j] * exp( -( (xe[i]-xc[j])**2 + (ye[i]-yc[j])**2 ) / (2* (sig0**2 + sig[j]**2) ) )
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


def compute_residual_stats(dfunc, c, sig, xc, yc, dims, base_level=0., square_c=True, compact_supp=True, resolution=1):
    """
    Computes the residual stats between appproximation and real data
    """

    _xe = np.linspace(0., 1., resolution*dims[0]+2)[1:-1]
    _ye = np.linspace(0., 1., resolution*dims[1]+2)[1:-1]
    len_xe = len(_xe); len_ye = len(_ye)
    Xe,Ye = np.meshgrid(_xe, _ye, sparse=False)
    xe = Xe.ravel(); ye = Ye.ravel()
    Nc = len(xc)
    Ne = len(xe)


    u = evaluate(c, sig, xe, ye, xc, yc, supp=supp, sig0=minsig) + base_level
    u = u.reshape(len_xe, len_ye)
    
    #Computing distance matrices
    #Dx = np.empty((Ne,Nc))
    #Dy = np.empty((Ne,Nc))
    #for k in range(Ne):
    #    Dx[k,:] = xe[k]-xc
    #    Dy[k,:] = ye[k]-yc
    
    #Computing the Phi matrix
    #if square_c: c = c**2
    #if compact_supp: phi_m = phi(Dx, Dy, sig.reshape(1,-1))
    #else: phi_m = phi(Dx, Dy, sig.reshape(1,-1), supp=0.)
    #u = np.dot(phi_m, c) + base_level

    residual = dfunc(_xe, _ye)-u
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
