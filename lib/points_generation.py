import ghalton
import numpy as np
import scipy.stats as st


def _inv_gaussian_kernel(kernlen=3, sig=0.1):
    """
    Returns a 2D Gaussian kernel array.
    """
    interval = (2*sig+1.)/(kernlen)
    x = np.linspace(-sig-interval/2., sig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel.max()-kernel


def boundary_generation(n_boundary):
    xb = []
    yb = []

    for val in np.linspace(0., 1., n_boundary+1)[0:-1]:
        xb.append(val)
        yb.append(0.)
    for val in np.linspace(0., 1., n_boundary+1)[0:-1]:
        xb.append(1.)
        yb.append(val)
    for val in np.linspace(0., 1., n_boundary+1)[::-1][:-1]:
        xb.append(val)
        yb.append(1.)
    for val in np.linspace(0., 1., n_boundary+1)[::-1][:-1]:
        xb.append(0.)
        yb.append(val)
    xb = np.asarray(xb)
    yb = np.asarray(yb)
    boundary_points = np.vstack([xb,yb]).T
    return boundary_points


def random_centers_generation(data, n_centers, cut_value_leq=None, cut_value_geq=None, power=5.):
    data = np.copy(data)
    if cut_value_leq is not None:
        mask = data <= cut_value_leq
    elif cut_value_geq is not None:
        mask = data >= cut_value_geq
    data **= power
    # fixed seed
    np.random.seed(0)
    # data dimensions
    m,n = data.shape
    
    # center points positions
    x = np.linspace(0., 1., data.shape[0]+2, endpoint=True)[1:-1]
    y = np.linspace(0., 1., data.shape[1]+2, endpoint=True)[1:-1]
    X,Y  = np.meshgrid(x,y)
    points_positions = np.vstack( [ X.ravel(), Y.ravel() ]).T
    
    # array with indexes of such centers
    points_indexes = np.arange(0,points_positions.shape[0])
    
    # array with probabilities of selection for each center
    #prob = np.zeros(m+2, n+2)
    #prob[1:m+1, 1:n+1] = (data/data.sum())
    if isinstance(mask, np.ndarray):
        data[mask] = 0.
        prob = data/data.sum()
    else:
        prob = data/data.sum()
    
    # convolution kernel
    #K = np.array([[0.5, 0.5, 0.5], [0.5, 0., 0.5], [0.5, 0.5, 0.5]])
    K = _inv_gaussian_kernel(kernlen=3, sig=3.)
    
    selected = []
    while len(selected)!=n_centers:
        sel = np.random.choice(points_indexes, size=1 , p=prob.ravel(), replace=False)[0]
        # border pixels can't be selected
        index0 = sel / m
        index1 = sel % n
        if index0==0 or index0==m-1 or index1==0 or index1==n-1: continue
        selected.append(sel)
        # update the pixel probabilities array
        prob[index0-1:index0+2, index1-1:index1+2] *= K
        #prob[index0, index1] = 0.
        prob /= prob.sum()
        
    return points_positions[selected]


def qrandom_centers_generation(dfunc, n_centers, base_level, ndim=2, get_size=50):
    # generating the sequencer
    sequencer = ghalton.Halton(ndim)

    points_positions = []
    n_selected = 0

    while True:
        points = sequencer.get(get_size)
        points_X, points_Y = zip(*points)
        for i in range(get_size):
            if dfunc(points_X[i], points_Y[i]) > base_level:
                points_positions.append(points[i])
                n_selected += 1
            if n_selected == n_centers:
                return np.asarray(points_positions)
