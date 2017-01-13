import sys
import numba
import numpy as np
import numpy.ma as ma
import scipy as sp
import numexpr as ne
from math import sqrt, exp
from scipy.interpolate import RegularGridInterpolator 

# ACALIB helper functions
sys.path.append('../../ACALIB/')
import acalib
from acalib import load_fits, standarize




@numba.jit('float64[:] (float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64)', nopython=True)
def u_eval(c, sig, xc, yc, xe, ye, support=5):
    m = len(xe)
    n = len(xc)
    ret = np.zeros(m)
    for i in range(m):
        for j in range(n):
            dist2 = (xe[i]-xc[j])**2 + (ye[i]-yc[j])**2
            if  dist2 > support**2 * sig[j]**2: continue
            ret[i] += c[j] * exp( -dist2 / (2* sig[j]**2 ) )
    return ret 


@numba.jit('float64[:,:] (float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64)', nopython=True)
def u_eval_full(c, sig, xc, yc, xe, ye, support=5):
    m = len(xe)
    n = len(xc)
    ret = np.zeros((6,m))
    for i in range(m):
        for j in range(n):
            dist2 = (xe[i]-xc[j])**2 + (ye[i]-yc[j])**2
            if  dist2 > support**2 * sig[j]**2: continue
            # common terms
            aux1 = (-1./(sig[j]**2))
            aux2 = aux1**2
            # u evaluation
            ret[0,i] += c[j] * exp( -dist2 / (2* sig[j]**2 ) )
            # ux evaluation
            ret[1,i] += aux1 * ret[0,i] * (xe[i]-xc[j])
            # uy evaluation
            ret[2,i] += aux1 * ret[0,i] * (ye[i]-yc[j])
            # uxy evaluation
            ret[3,i] += aux2 * ret[0,i] * (xe[i]-xc[j])*(ye[i]-yc[j])
            # uxx evaluation (REVISAR ESTO)
            ret[4,i] += aux2 * ret[0,i] * ((xe[i]-xc[j])**2 - sig[j]**2)
            # uyy evaluation (REVISAR ESTO)
            ret[5,i] += aux2 * ret[0,i] * ((ye[i]-yc[j])**2 - sig[j]**2)
    return ret



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
    
    # in case NaN values exist on cube
    mask = np.isnan(data)
    if np.any(mask): data = ma.masked_array(data, mask=mask)

    # map to 0-1 intensity range
    data -= data.min()
    data /= data.max()
    
    if data.shape[0]==1:
        data = np.ascontiguousarray(data[0])
        if np.any(mask): 
            mask = np.ascontiguousarray(mask[0])
            data = ma.masked_array(data, mask=mask)
        # generating the data function
        x = np.linspace(0., 1., data.shape[0]+2, endpoint=True)[1:-1]
        y = np.linspace(0., 1., data.shape[1]+2, endpoint=True)[1:-1]
        dfunc = RegularGridInterpolator((x,y), data, method='linear', bounds_error=False, fill_value=0.)
        return x,y,data,dfunc

    else:
        # generating the data function
        x = np.linspace(0., 1., data.shape[0]+2, endpoint=True)[1:-1]
        y = np.linspace(0., 1., data.shape[1]+2, endpoint=True)[1:-1]
        z = np.linspace(0., 1., data.shape[2]+2, endpoint=True)[1:-1]
        dfunc = RegularGridInterpolator((x, y, z), data, method='linear', bounds_error=False, fill_value=0.)
        return x, y, z, data, dfunc


def logistic(x):
    return 1. / (1. + np.exp(-x))


def logit(x):
    mask0 = x==0.
    mask1 = x==1.
    mask01 = np.logical_and(~mask0, ~mask1)
    res = np.empty(x.shape[0])
    res[mask0] = -np.inf
    res[mask1] = np.inf
    res[mask01] = np.log(x[mask01] / (1-x[mask01]))
    return res


def mean_min_dist(points1, points2):
    x1 = points1[:,0]; y1 = points1[:,1]
    x2 = points2[:,0]; y2 = points2[:,1]
    M = points1.shape[0]
    N = points2.shape[0]
    Dx = np.empty((M,N))
    Dy = np.empty((M,N))
    for k in range(M):
        Dx[k] = x1[k] - x2
        Dy[k] = y1[k] - y2
    D = np.sqrt(Dx**2 + Dy**2)
    return np.mean( np.min(D, axis=1) )