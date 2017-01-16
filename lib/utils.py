import sys
import numba
import numpy as np
import numpy.ma as ma
import scipy as sp
import numexpr as ne
from math import sqrt, exp
from scipy.interpolate import RegularGridInterpolator
from astropy.io import fits

# ACALIB helper functions
#sys.path.append('../../ACALIB/')
#import acalib
#from acalib import load_fits, standarize




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


def snr_estimation(data, noise=None, points=1000, full_output=False):
    """
    Heurustic that uses the inflexion point of the thresholded RMS to estimate where signal is dominant w.r.t. noise
    
    Parameters
    ---------- 
    data : (M,N,Z) numpy.ndarray or numpy.ma.MaskedArray


    noise : float (default=None)
        Noise level, if not given will use rms of the data.
    
    points : (default=1000)

    full_output : boolean (default=False)
        Gives verbose results if True

    Returns
    --------

    "Signal to Noise Radio" value
    
    """
    if noise is None:
        noise = estimate_rms(data)
    x = []
    y = []
    n = []
    sdata = data[data > noise]
    for i in range(1, int(points)):
        val = 1.0 + 2.0 * i / points
        sdata = sdata[sdata > val * noise]
        if sdata.size < 2:
            break
        n.append(sdata.size)
        yval = sdata.mean() / noise
        x.append(val)
        y.append(yval)
    y = np.array(y)
    v = y[1:] - y[0:-1]
    p = v.argmax() + 1
    snrlimit = x[p]
    if full_output == True:
        return snrlimit, noise, x, y, v, n, p
    return snrlimit



def build_dist_matrix(points, inf=False):
    """
    Builds a distance matrix from points array.
    It returns a (n_points, n_points) distance matrix. 

    points: NumPy array with shape (n_points, 2) 
    """
    m,n = points.shape
    D = np.empty((m,m))
    for i in range(m):
        for j in range(m):
            if inf and i==j: 
                D[i,j] = np.inf
                continue 
            D[i,j] = np.linalg.norm(points[i]-points[j], ord=2)
    return D


def load_data(fits_path):
    hdulist = fits.open(fits_path)
    data = hdulist[0].data
    # droping out the stokes dimension
    data = np.ascontiguousarray(data[0])
    
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
        return x,y,z,data,dfunc


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

