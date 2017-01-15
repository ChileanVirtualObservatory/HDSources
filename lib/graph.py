import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utils import u_eval



def image_plot(data, title='FITS image'):
    plt.figure(figsize=(6,6))
    im = plt.imshow(data, cmap=plt.cm.afmhot)
    plt.title(title)
    plt.axis('off')
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.show()


def thresholded_image_plot(data, level):
    plt.figure(figsize=(6,6))
    ax = plt.gca()
    _data = np.zeros(data.shape)
    mask = data > level
    _data[mask] = data[mask]
    im = ax.imshow(_data, cmap=plt.cm.afmhot)
    plt.title('Thresholded data')
    plt.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.08)
    plt.colorbar(im, cax=cax)
    plt.show()


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



def solution_plot(dfunc, c, sig, xc, yc, dims, base_level=0., mask=None, 
                 resolution=1, title=None, support=5.):
    _xe = np.linspace(0., 1., resolution*dims[0]+2)[1:-1]
    _ye = np.linspace(0., 1., resolution*dims[1]+2)[1:-1]
    len_xe = len(_xe); len_ye = len(_ye)
    Xe,Ye = np.meshgrid(_xe, _ye, sparse=False, indexing='ij')
    xe = Xe.ravel(); ye = Ye.ravel()
    points = np.vstack([xe,ye]).T

    # approximation
    u = u_eval(c, sig, xc, yc, xe, ye, support=support) + base_level
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
    im = ax.imshow(f, vmin=0., vmax=1., cmap=plt.cm.afmhot)
    plt.title('Original')
    plt.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    
    # approximated solution plot
    plt.subplot(1,3,2)
    ax = plt.gca()
    im = ax.imshow(u, vmin=0., vmax=1., cmap=plt.cm.afmhot)
    plt.title('Solution')
    plt.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    
    # residual plot
    plt.subplot(1,3,3)
    ax = plt.gca()
    im = ax.imshow(res, vmin=0., vmax=1., cmap=plt.cm.afmhot)
    plt.title('Residual')
    plt.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.show()



def params_plot(c, sig, xc, yc, remove_outlier=True):
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
    plt.scatter(yc_, xc_ , c=c_)
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
    plt.scatter(yc_, xc_, c=sig_**2)
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.show()
 


def params_distribution_plot(c, sig, remove_outlier=True):
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
        plt.imshow(data, cmap=plt.cm.afmhot)
        plt.scatter(center_points[:,1]*y_scale, center_points[:,0]*x_scale, c='#7272e8', s=5, label='center')
        if title is not None: plt.title(title) 
        else: plt.title('Center points')
        plt.axis('off')
    elif (center_points is None) and (collocation_points is not None):
        plt.figure(figsize=(8,8))
        plt.imshow(data, cmap=plt.cm.afmhot)
        plt.scatter(collocation_points[:,1]*y_scale, collocation_points[:,0]*x_scale, c='#7272e8', s=5, label='collocation')
        if title is not None: plt.title(title)
        else: plt.title('Collocation points')
        plt.axis('off')
    elif (center_points is not None) and (collocation_points is not None):
        fig = plt.figure(figsize=(20,15))
        ax1 = fig.add_subplot(121)
        ax1.imshow(data, cmap=plt.cm.afmhot)
        ax1.scatter(center_points[:,1]*y_scale, center_points[:,0]*x_scale, c='#7272e8', s=5, label='center')
        if title is not None: plt.title(title)
        else: ax1.set_title('Center points')
        ax1.axis('off')
        ax2 = fig.add_subplot(122)
        ax2.imshow(data, cmap=plt.cm.afmhot)
        ax2.scatter(collocation_points[:,1]*y_scale, collocation_points[:,0]*x_scale, c='#7272e8', s=5, label='collocation')
        if title is not None: plt.title(title)
        else: ax2.set_title('Collocation points')
        ax2.axis('off')
    if (boundary_points is not None) and len(boundary_points[:,0])!=0:
        plt.figure(figsize=(8,8))
        plt.imshow(data, cmap=plt.cm.afmhot)
        plt.scatter(boundary_points[:,1]*y_scale, boundary_points[:,0]*x_scale, c='#7272e8', s=5, label="boundary")
        plt.axis('off')
    #plt.colorbar(im, cax=cax)
    #fig.legend(bbox_to_anchor=(1.2, 1.0))
    plt.show()

    
def components_plot(elm, components_dict, n_comp, n_levels=5):
    # get all the (mapped) parameters
    xc, yc, c, sig = elm.get_params_mapped()
    
    # generating the evaluation points
    _xe = np.linspace(0., 1., elm.dims[0]+2)[1:-1]
    _ye = np.linspace(0., 1., elm.dims[1]+2)[1:-1]
    len_xe = len(_xe); len_ye = len(_ye)
    Xe,Ye = np.meshgrid(_xe, _ye, sparse=False, indexing='ij')
    xe = Xe.ravel(); ye = Ye.ravel()  
    
    plt.figure(figsize=(10,10))
    plt.title('{0} components solution'.format(n_comp))
    plt.axis('off')
    ax = plt.subplot(1,1,1)
    ax.imshow(elm.data, cmap=plt.cm.afmhot)
    color = plt.cm.rainbow(np.linspace(0., 1., n_comp))
    levels = np.linspace(1.05*elm.base_level, 0.95, n_levels)
    
    for i,indexes in enumerate(components_dict[n_comp]):
        _xc = xc[indexes]
        _yc = yc[indexes]
        _c = c[indexes]
        _sig = sig[indexes]
        u = u_eval(_c, _sig, _xc, _yc, xe, ye, support=elm.support) + elm.base_level
        _u = u.reshape(len_xe, len_ye)
        
        ax.contour(_u, levels=levels, colors=[color[i]])
        #plt.subplot(n_comp,1, i+1)
        #ax = plt.gca()
        #im = ax.imshow(_u, vmin=0., vmax=1.)
        #plt.axis('off')
        #divider = make_axes_locatable(ax)
        #cax = divider.append_axes("right", size="5%", pad=0.05)
        #plt.colorbar(im, cax=cax)
    plt.show()
    

########################################################
# 3D only functions
########################################################

def points_plot3D(points, title=None):
    x = points[:,0]
    y = points[:,1]
    z = points[:,2]

    # visualization of points
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c='r', marker='o', s=7)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.title(title)
    plt.show()



def slices_plot(data, slc):
    plt.figure(figsize=(5,5))
    im = plt.imshow(data[slc], vmin=0, vmax=1., cmap=plt.cm.gray)
    plt.title('3D cube at slice: {0}'.format(slc))
    plt.axis('off')
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.show()
  

    
def comparative_slices_plot(data1, data2, slc):
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    im = plt.imshow(data1[slc], vmin=0, vmax=1., cmap=plt.cm.gray)
    plt.title('3D original cube at slice: {0}'.format(slc))
    plt.axis('off')
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.subplot(1,2,2)
    im = plt.imshow(data2[slc], vmin=0, vmax=1., cmap=plt.cm.gray)
    plt.title('3D approximated cube at slice: {0}'.format(slc))
    plt.axis('off')
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.show()