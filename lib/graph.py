import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utils import phi, u_eval


##################################################################
# GLOBAL VARIABLES
##################################################################
supp = 5.      # gaussians support
minsig = 0.001 # guassians minimal broadening


def plotter(dfunc, c, sig, xc, resolution=10, title=None):
    """
    Helper function to visualize the quality of the solution
    """
    _xe = np.linspace(0., 1., 10*N, endpoint=True)
    _Dx = np.empty((10*N,N))
    for k in range(10*N):
        _Dx[k,:] = (_xe[k] - xc)

    phi_m = phi(_Dx, sig)
    u = np.dot(phi_m, c)
    plt.figure(figsize=(10,6))
    plt.plot(_xe, u, 'r-', label='Solution')
    plt.plot(x_, dfunc(x_), 'b--', label='Data')
    plt.plot(xe, dfunc(xe), 'go', label='Evaluation points')
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.3, 1.0))
    plt.show()


def solution_plot(dfunc, c, sig, xc, yc, dims, base_level=0., square_c=True, mask=None, 
                 resolution=1, title=None, compact_supp=True):
    _xe = np.linspace(0., 1., resolution*dims[0]+2)[1:-1]
    _ye = np.linspace(0., 1., resolution*dims[1]+2)[1:-1]
    len_xe = len(_xe); len_ye = len(_ye)
    Xe,Ye = np.meshgrid(_xe, _ye, sparse=False)
    xe = Xe.ravel(); ye = Ye.ravel()
    points = np.vstack([xe,ye]).T
    Nc = len(xc)
    Ne = len(xe)

    # approximation
    if square_c: c = c**2
    u = u_eval(c, sig, xe, ye, xc, yc, supp=supp, sig0=minsig) + base_level
    u = u.reshape(len_xe, len_ye)

    # real data
    f = dfunc(points).reshape(dims)

    # residual
    res = f-u+base_level


    # unusable pixels are fixed to 0
    if mask is not None: 
        u[~mask] = 0.
        f[~mask] = 0.
        res[~mask] = 0.


    # original data plot
    plt.figure(figsize=(18,12))
    plt.subplot(1,3,1)
    ax = plt.gca()
    im = ax.imshow(f, vmin=0., vmax=1.)
    plt.title('Original')
    plt.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    
    # approximated solution plot
    plt.subplot(1,3,2)
    ax = plt.gca()
    im = ax.imshow(u, vmin=0., vmax=1.)
    plt.title('Solution')
    plt.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    
    # residual plot
    plt.subplot(1,3,3)
    ax = plt.gca()
    im = ax.imshow(res, vmin=0., vmax=1.)
    plt.title('Residual')
    plt.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.show()


def params_plot(c, sig, xc, yc, square_c=True, remove_outlier=True):
    if square_c: c = c**2
    if remove_outlier:
        # just keeping values less than 10 times the mean
        c_med = np.median(c)
        mask1 = c < 10.*c_med
        c_ = c[mask1]
        xc_ = xc[mask1]; yc_ = yc[mask1]
    plt.figure(figsize=(17,7))
    plt.subplot(1,2,1)
    plt.title('Plot of c parameters')
    plt.xlim(-0.01,1.01)
    plt.ylim(-0.01,1.01)
    plt.scatter(xc_, yc_, c=c_)
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.subplot(1,2,2)
    if remove_outlier:
        # just keeping values less than 10 times the mean
        sig2_med = np.median(sig**2)
        mask2 = sig**2 < 10.*sig2_med
        sig_ = sig[mask2]
        xc_ = xc[mask2]; yc_ = yc[mask2]
    plt.title('Plot of sig^2 parameters')
    plt.xlim(-0.01,1.01)
    plt.ylim(-0.01,1.01)
    plt.scatter(xc_, yc_, c=sig_**2)
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.show()
 

def params_distribution_plot(c, sig, square_c=True, remove_outlier=True):
    if square_c: c = c**2
    if remove_outlier:
        # just keeping values less than 10 times the median
        c_med = np.median(c)
        sig2_med = np.median(sig**2)
        mask1 = c < 10.*c_med
        c = c[mask1]
        mask2 = sig**2 < 10.*sig2_med
        sig = sig[mask2]
    plt.figure(figsize=(14,7))
    plt.subplot(1,2,1)
    plt.title('C distribution')
    plt.hist(c, color='grey', bins=10)
    plt.subplot(1,2,2)
    plt.title('Sig^2 distribution')
    plt.hist(sig**2, color='grey', bins=10)
    plt.show()


def residual_plot(residual_variance, residual_entropy, residual_rms, iter_list):
    plt.figure(figsize=(12,5))
    plt.subplot(1,3,1)
    plt.xlim(0, iter_list[-1]+iter_list[0])
    plt.plot(iter_list, residual_rms, 'go-')
    plt.title('Residual RMS')        
    plt.subplot(1,3,2)
    plt.xlim(0, iter_list[-1]+iter_list[0])
    plt.plot(iter_list, residual_variance, 'bo-')
    plt.title('Residual variance')
    plt.subplot(1,3,3)
    plt.xlim(0, iter_list[-1]+iter_list[0])
    plt.plot(iter_list, residual_entropy, 'ro-')
    plt.title('Residual entropy')
    plt.show()


def points_plot(data, center_points=None, collocation_points=None, boundary_points=None, title=None):
    x_scale = data.shape[0]-1
    y_scale = data.shape[1]-1
    if (center_points is not None) and (collocation_points is None):
        plt.figure(figsize=(8,8))
        plt.imshow(data)
        plt.scatter(center_points[:,0]*x_scale, center_points[:,1]*y_scale, c='r', s=5, label='center')
        if title is not None: plt.title(title) 
        else: plt.title('Center points')
        plt.axis('off')
    elif (center_points is None) and (collocation_points is not None):
        plt.figure(figsize=(8,8))
        plt.imshow(data)
        plt.scatter(collocation_points[:,0]*x_scale, collocation_points[:,1]*y_scale, c='g', s=5, label='collocation')
        if title is not None: plt.title(title)
        else: plt.title('Collocation points')
        plt.axis('off')
    elif (center_points is not None) and (collocation_points is not None):
        fig = plt.figure(figsize=(20,15))
        ax1 = fig.add_subplot(121)
        ax1.imshow(data)
        ax1.scatter(center_points[:,0]*x_scale, center_points[:,1]*y_scale, c='r', s=5, label='center')
        if title is not None: plt.title(title)
        else: ax1.set_title('Center points')
        ax1.axis('off')
        ax2 = fig.add_subplot(122)
        ax2.imshow(data)
        ax2.scatter(collocation_points[:,0]*x_scale, collocation_points[:,1]*y_scale, c='g', s=5, label='collocation')
        if title is not None: plt.title(title)
        else: ax2.set_title('Collocation points')
        ax2.axis('off')
    if (boundary_points is not None) and len(boundary_points[:,0])!=0:
        plt.figure(figsize=(8,8))
        plt.imshow(data)
        plt.scatter(boundary_points[:,0]*x_scale, boundary_points[:,1]*y_scale, c='y', s=5, label="boundary")
        plt.axis('off')
    #plt.colorbar(im, cax=cax)
    #fig.legend(bbox_to_anchor=(1.2, 1.0))
    plt.show()
