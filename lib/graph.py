import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utils import u_eval
from utils3D import u_eval as u_eval3D
from utils3D import compute_solution
from gmr import isd_diss_full
from points_generation import _boundary_map



def image_plot(data, title='FITS image'):
    plt.figure(figsize=(10,10))
    im = plt.imshow(data, cmap=plt.cm.gray_r, interpolation=None)
    plt.title(title)
    #plt.axis('off')
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.show()


def thresholded_image_plot(data, level):
    plt.figure(figsize=(10,10))
    ax = plt.gca()
    _data = np.zeros(data.shape)
    mask = data > level
    _data[mask] = data[mask]
    im = ax.imshow(_data, cmap=plt.cm.gray_r)
    plt.title('Thresholded data at: {0}'.format(level))
    #plt.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.08)
    plt.colorbar(im, cax=cax)
    plt.show()
    print("{0} usable pixels out of {1}".format(np.sum(mask), data.shape[0]*data.shape[1]))


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
    #res = f-u+base_level
    res = f-u

    # unusable pixels are fixed to 0
    if mask is not None: 
        u[~mask] = 0.
        f[~mask] = 0.
        res[~mask] = 0.

    # original data plot
    plt.figure(figsize=(18,12))
    plt.subplot(1,3,1)
    ax = plt.gca()
    im = ax.imshow(f, vmin=0., vmax=1., cmap=plt.cm.gray_r)
    plt.title('Original')
    plt.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    
    # approximated solution plot
    plt.subplot(1,3,2)
    ax = plt.gca()
    im = ax.imshow(u, vmin=0., vmax=1., cmap=plt.cm.gray_r)
    plt.title('Solution')
    plt.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    
    # residual plot
    plt.subplot(1,3,3)
    ax = plt.gca()
    im = ax.imshow(res, vmin=0., vmax=1., cmap=plt.cm.gray_r)
    plt.title('Residual')
    plt.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.show()



def params_plot(c, sig, xc, yc, remove_outlier=False):
    if remove_outlier:
        # just keeping values less than 10 times the mean
        c_med = np.median(c)
        mask1 = c < 10.*c_med
        c = c[mask1]
        xc = xc[mask1]; yc = yc[mask1]
    plt.figure(figsize=(17,7))
    plt.subplot(1,2,1)
    plt.title('Plot of c parameters')
    plt.xlim(-0.01,1.01)
    plt.ylim(-0.01,1.01)
    plt.scatter(yc, xc , c=c)
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.subplot(1,2,2)
    if remove_outlier:
        # just keeping values less than 10 times the mean
        sig2_med = np.median(sig**2)
        mask2 = sig**2 < 10.*sig2_med
        sig = sig[mask2]
        xc = xc[mask2]; yc = yc[mask2]
    plt.title('Plot of sig^2 parameters')
    plt.xlim(-0.01,1.01)
    plt.ylim(-0.01,1.01)
    plt.scatter(yc, xc, c=sig**2)
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.show()
 


def params_distribution_plot(c, sig, remove_outlier=False):
    if remove_outlier:
        # just keeping values less than 10 times the median
        c_med = np.median(c)
        sig2_med = np.median(sig**2)
        mask1 = c < 10.*c_med
        c = c[mask1]
        mask2 = sig**2 < 10.*sig2_med
        sig = sig[mask2]
    plt.figure(figsize=(16,8))
    plt.subplot(1,2,1)
    plt.title('C distribution')
    plt.hist(c, bins=10, facecolor='seagreen', edgecolor='black', lw=2)
    plt.subplot(1,2,2)
    plt.title('Sig^2 distribution')
    plt.hist(sig**2, bins=10, facecolor='peru', edgecolor='black', lw=2)
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


    
def points_plot(data, center_points=None, collocation_points=None, boundary_points=None, title=None, save_path=None):
    x_scale = data.shape[0]-1
    y_scale = data.shape[1]-1
    if (center_points is not None) and (collocation_points is None):
        plt.figure(figsize=(10,10))
        plt.imshow(data, cmap=plt.cm.gray_r)
        plt.scatter(center_points[:,1]*y_scale, center_points[:,0]*x_scale, c='magenta', s=12, label='center')
        plt.grid()
        plt.tick_params(axis='both', which='major', labelsize=20)
        if title is not None: plt.title(title) 
        else: plt.title('Center points')
        #plt.axis('off')
    elif (center_points is None) and (collocation_points is not None):
        plt.figure(figsize=(10,10))
        plt.imshow(data, cmap=plt.cm.gray_r)
        plt.scatter(collocation_points[:,1]*y_scale, collocation_points[:,0]*x_scale, c='green', s=12, label='collocation')
        if title is not None: plt.title(title)
        else: plt.title('Collocation points')
        #plt.axis('off')
    elif (center_points is not None) and (collocation_points is not None):
        fig = plt.figure(figsize=(20,15))
        ax1 = fig.add_subplot(121)
        ax1.imshow(data, cmap=plt.cm.gray_r)
        ax1.scatter(center_points[:,1]*y_scale, center_points[:,0]*x_scale, c='green', s=12, label='center')
        if title is not None: plt.title(title)
        else: ax1.set_title('Center points')
        #ax1.axis('off')
        ax2 = fig.add_subplot(122)
        ax2.imshow(data, cmap=plt.cm.gray_r)
        ax2.scatter(collocation_points[:,1]*y_scale, collocation_points[:,0]*x_scale, c='green', s=12, label='collocation')
        if title is not None: plt.title(title)
        else: ax2.set_title('Collocation points')
        #ax2.axis('off')
    if (boundary_points is not None) and len(boundary_points[:,0])!=0:
        plt.figure(figsize=(10,10))
        plt.imshow(data, cmap=plt.cm.gray_r)
        plt.scatter(boundary_points[:,1]*y_scale, boundary_points[:,0]*x_scale, c='green', s=12, label="boundary")
        #plt.axis('off')
    #plt.colorbar(im, cax=cax)
    #fig.legend(bbox_to_anchor=(1.2, 1.0))
    if save_path is not None:
        plt.savefig(save_path, format='eps', dpi=50)
    plt.show()

    
def components_plot(elm, components_dict, n_comp, n_levels=1, show_title=False, show_isd=False, save_path=None):
    # get all the (mapped) parameters
    xc, yc, c, sig = elm.get_params_mapped()
    
    # generating the evaluation points
    _xe = np.linspace(0., 1., elm.dims[0]+2)[1:-1]
    _ye = np.linspace(0., 1., elm.dims[1]+2)[1:-1]
    len_xe = len(_xe); len_ye = len(_ye)
    Xe,Ye = np.meshgrid(_xe, _ye, sparse=False, indexing='ij')
    xe = Xe.ravel(); ye = Ye.ravel()  
    
    plt.figure(figsize=(10,10))
    plt.tick_params(axis='both', which='major', labelsize=1)
    plt.grid()
    if show_title: plt.title('{0} components solution'.format(n_comp))

    ax = plt.subplot(1,1,1)
    ax.imshow(elm.data, cmap=plt.cm.gray_r)

    # generating the color of sources
    maxclump = 20
    color = plt.cm.rainbow(np.linspace(0., 1., maxclump))
    np.random.seed(1); np.random.shuffle(color)
    color = color[0:n_comp]

    if n_levels==1:
        levels = [1.05*elm.base_level]
    else:
        levels = np.linspace(1.05*elm.base_level, 0.95, n_levels)

    if show_isd:
        # putting parameters in the correct format
        w = elm.get_w()
        mu = np.vstack([xc,yc]).T
        Sig = np.zeros((len(w),2,2))
        Sig[:,0,0] = sig; Sig[:,1,1] = sig

    for i,indexes in enumerate(components_dict[n_comp]):
        _xc = xc[indexes]
        _yc = yc[indexes]
        _c = c[indexes]
        _sig = sig[indexes]
        u = u_eval(_c, _sig, _xc, _yc, xe, ye, support=elm.support) + elm.base_level
        _u = u.reshape(len_xe, len_ye)

        if show_isd:
            _isd = isd_diss_full(w[indexes], mu[indexes], Sig[indexes])
            cs = ax.contour(_u, levels=levels, colors=[color[i]], linewidths=4)
            cs.collections[0].set_label('ISD: {0}'.format(_isd))
        else:
            ax.contour(_u, levels=levels, colors=[color[i]], linewidths=4)
    if show_isd: plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
    if save_path is not None:
        plt.savefig(save_path, format='eps', dpi=150)
    plt.show()



def caa_show(data, caa, save_path=None):
    bd_map = _boundary_map(caa)
    colors = plt.cm.rainbow(np.linspace(0., 1., bd_map.max()))
    
    cmap = plt.cm.gray_r
    norm = plt.Normalize(data.min(), data.max())
    rgba = cmap(norm(data.T))
    
    m,n = data.shape
    for i in range(m):
        for j in range(n):
            if bd_map[i,j]==0: continue
            rgba[i,j,:] = colors[bd_map[i,j]-1]

    plt.figure(figsize=(10,10))
    plt.tick_params(axis='both', which='major', labelsize=0)
    plt.grid()
    plt.imshow(rgba)
    if save_path is not None:
        plt.savefig(save_path, format='eps', dpi=150)
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
    im = plt.imshow(data[slc], vmin=0, vmax=1., cmap=plt.cm.gray_r)
    plt.title('3D cube at slice: {0}'.format(slc))
    plt.axis('off')
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.show()
  

    

def solution_plot3D(elm):
    # original (stacked) data
    _data = elm.data.sum(axis=0)
    dmin = _data.min(); dmax = _data.max()
    _data -= dmin; _data /= dmax

    # approximated solution
    xc, yc, zc, c, sig = elm.get_params_mapped()
    u = compute_solution(c, sig, xc, yc, zc, elm.dims, base_level=elm.base_level)
    _u = u.sum(axis=0)
    _u -= dmin; _u /= dmax

    # residual
    res = _data-_u+_u.min()


    # original data plot
    plt.figure(figsize=(18,12))
    plt.subplot(1,3,1)
    ax = plt.gca()
    im = ax.imshow(_data, vmin=0., vmax=1., cmap=plt.cm.gray_r)
    plt.title('Original')
    plt.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    
    # approximated solution plot
    plt.subplot(1,3,2)
    ax = plt.gca()
    im = ax.imshow(_u, vmin=0., vmax=1., cmap=plt.cm.gray_r)
    plt.title('Solution')
    plt.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    
    # residual plot
    plt.subplot(1,3,3)
    ax = plt.gca()
    im = ax.imshow(res, vmin=0., vmax=1., cmap=plt.cm.gray_r)
    plt.title('Residual')
    plt.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.show()


def comparative_slices_plot(data1, data2, slc):
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    im = plt.imshow(data1[slc], vmin=0, vmax=1., cmap=plt.cm.gray_r)
    plt.title('3D original cube at slice: {0}'.format(slc))
    plt.axis('off')
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.subplot(1,2,2)
    im = plt.imshow(data2[slc], vmin=0, vmax=1., cmap=plt.cm.gray_r)
    plt.title('3D approximated cube at slice: {0}'.format(slc))
    plt.axis('off')
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.show()


def components_plot3D(elm, components_dict, n_comp, n_levels=10):
    # get all the (mapped) parameters
    xc, yc, zc, c, sig = elm.get_params_mapped()

    # generating the evaluation points
    _xe = np.linspace(0., 1., elm.dims[0]+2)[1:-1]
    _ye = np.linspace(0., 1., elm.dims[1]+2)[1:-1]
    _ze = np.linspace(0., 1., elm.dims[2]+2)[1:-1]
    len_xe = len(_xe); len_ye = len(_ye); len_ze = len(_ze)
    Xe,Ye,Ze = np.meshgrid(_xe, _ye, _ze, sparse=False, indexing='ij')
    xe = Xe.ravel(); ye = Ye.ravel(); ze = Ze.ravel()  

    plt.figure(figsize=(8,8))
    plt.title('{0} components solution'.format(n_comp))
    plt.axis('off')
    ax = plt.subplot(1,1,1)

    # stacked data, mapping to [0,1] and display 
    _data = elm.data.sum(axis=0)
    dmin = _data.min(); dmax = _data.max()
    _data -= dmin
    _data /= dmax
    ax.imshow(_data, cmap=plt.cm.afmhot)

    # contours configuration
    minval = ((elm.base_level*elm.dims[0]) - dmin) / dmax
    color = plt.cm.rainbow(np.linspace(0., 1., n_comp))
    levels = np.linspace(minval+0.01, 0.95, n_levels)

    for i,indexes in enumerate(components_dict[n_comp]):
        _xc = xc[indexes]
        _yc = yc[indexes]
        _zc = zc[indexes]
        _c = c[indexes]
        _sig = sig[indexes]
        u = u_eval3D(_c, _sig, _xc, _yc, _zc, xe, ye, ze, support=elm.support) + elm.base_level
        _u = u.reshape(len_xe, len_ye, len_ze).sum(axis=0)
        _u -= dmin; _u /= dmax
        ax.contour(_u, levels=levels, colors=[color[i]])
    plt.show()


def components_plot3D_(elm, components_dict, n_comp):
    # get all the (mapped) parameters
    xc, yc, zc, c, sig = elm.get_params_mapped()

    # generating the evaluation points
    _xe = np.linspace(0., 1., elm.dims[0]+2)[1:-1]
    _ye = np.linspace(0., 1., elm.dims[1]+2)[1:-1]
    _ze = np.linspace(0., 1., elm.dims[2]+2)[1:-1]
    len_xe = len(_xe); len_ye = len(_ye); len_ze = len(_ze)
    Xe,Ye,Ze = np.meshgrid(_xe, _ye, _ze, sparse=False, indexing='ij')
    xe = Xe.ravel(); ye = Ye.ravel(); ze = Ze.ravel()  

    clump_map = np.empty(elm.dims)

    # stacked data, mapping to [0,1] and display 
    base_level = elm.base_level

    # contours configuration
    color_index = 20.

    for i,indexes in enumerate(components_dict[n_comp]):
        _xc = xc[indexes]
        _yc = yc[indexes]
        _zc = zc[indexes]
        _c = c[indexes]
        _sig = sig[indexes]
        u = u_eval3D(_c, _sig, _xc, _yc, _zc, xe, ye, ze, support=elm.support) + elm.base_level
        _u = u.reshape(len_xe, len_ye, len_ze)
        clump_map[_u > elm.base_level+0.01] = color_index
        color_index += 20.
    return clump_map


def _stat_plot(x_var, r_stats, stat, x_label='', loglog=False, n=5, slope=None, name=None):
    """
    Function to plot a single residual plot for a single image
    """
    stats = {'rms':0, 'var':1, 'fadd':2, 'flost':3, 'psiint':4, 'epix':5, 'sharp':6}

    y_label = ['RMS', 'Variance', 'Flux addition', 'Flux lost', \
               'Psi1 int', 'Excedeed pixels', 'Sharpness', 'Psi2 int']
    # unpacking the values
    r_stats_list = []
    r_stats_list.append( np.array([rms for (_,_,rms,_,_,_,_,_,_) in r_stats]) )
    r_stats_list.append( np.array([var for (var,_,_,_,_,_,_,_,_) in r_stats]) )
    #r_stats_list.append( np.array([entr for (_,entr,_,_,_,_,_,_,_) in r_stats]) )
    r_stats_list.append( np.array([flux for (_,_,_,flux,_,_,_,_,_) in r_stats]) )
    r_stats_list.append( np.array([flux for (_,_,_,_,flux,_,_,_,_) in r_stats]) )
    r_stats_list.append( np.array([psi1 for (_,_,_,_,_,psi1,_,_,_) in r_stats]) )
    r_stats_list.append( np.array([npix for (_,_,_,_,_,_,npix,_,_) in r_stats]) )
    r_stats_list.append( np.array([sharp for (_,_,_,_,_,_,_,sharp,_) in r_stats]) )
    r_stats_list.append( np.array([psi2 for (_,_,_,_,_,_,_,_,psi2) in r_stats]) )

    colors = plt.cm.rainbow(np.linspace(0., 1., len(r_stats_list)))

    i = stats[stat]
    r_stat = r_stats_list[i]
    fig = plt.figure(figsize=(17,6))
    fig.subplots_adjust(wspace=0.25)
    plt.subplot(1,2,1)
    plt.plot(x_var, r_stat, color=colors[i], marker='o')
    plt.grid()
    plt.xlabel(x_label, fontsize=20)
    plt.ylabel(y_label[i], fontsize=20)
    plt.subplot(1,2,2)
    if loglog:
        plt.loglog(x_var, r_stat, color=colors[i], marker='o')
        plt.grid()
        plt.xlabel(x_label, fontsize=20)
        plt.ylabel(y_label[i], fontsize=20)
        line = (r_stat[1]/x_var[1]**(slope)) * x_var**(slope)
        plt.plot(x_var, line, color='k', label='slope={0}'.format(slope))
        plt.legend(bbox_to_anchor=(1.3, 1.0))
    else:
        plt.semilogy(x_var, r_stat, color=colors[i], marker='o')
        plt.grid()
        plt.xlabel(x_label, fontsize=20)
        plt.ylabel(y_label[i], fontsize=20)

    if name is not None: plt.savefig(name, format='eps', dpi=1000)
    plt.show()


def stat_plots(x_var, y_list, labels, xlabel=None, ylabel=None, save_name=None, legend=False):
    """
    Function to plot a single residual stat for multiple images
    """
    plt.figure(figsize=(7,4))
    colors = plt.cm.rainbow(np.linspace(0., 1., 100))
    np.random.seed(0)
    np.random.shuffle(colors)
    colors = colors[0:len(y_list)]
    for i,y_var in enumerate(y_list):
        plt.plot(x_var, y_var, c=colors[i], marker='o', label=labels[i])
    if xlabel is not None: plt.xlabel(xlabel, fontsize=30)
    if ylabel is not None: plt.ylabel(ylabel, fontsize=25)
    plt.grid()
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tight_layout()
    if legend: plt.legend(loc='best', prop={'size':20})
    if save_name is not None:
        plt.savefig(save_name, format='eps', dpi=1000)
    plt.show()


def all_stats_plot(x_var, r_stats, x_label='', loglog=False, n=5, slope=None, name=None):
    """
    Function to plot all the residual stats for a single image
    """
    y_label = ['RMS', 'Flux addition', 'Flux lost', 'Sharpness']
    # unpacking the values
    r_stats_list = []
    r_stats_list.append( np.array([rms for (_,_,rms,_,_,_,_,_,_) in r_stats]) )
    #r_stats_list.append( np.array([var for (var,_,_,_,_,_,_,_,_) in r_stats]) )
    #r_stats_list.append( np.array([entr for (_,entr,_,_,_,_,_,_,_) in r_stats]) )
    r_stats_list.append( np.array([flux for (_,_,_,flux,_,_,_,_,_) in r_stats]) )
    r_stats_list.append( np.array([flux for (_,_,_,_,flux,_,_,_,_) in r_stats]) )
    #r_stats_list.append( np.array([psi1 for (_,_,_,_,_,psi1,_,_,_) in r_stats]) )
    #r_stats_list.append( np.array([npix for (_,_,_,_,_,_,npix,_,_) in r_stats]) )
    r_stats_list.append( np.array([sharp for (_,_,_,_,_,_,_,sharp,_) in r_stats]) )
    #r_stats_list.append( np.array([psi2 for (_,_,_,_,_,_,_,_,psi2) in r_stats]) )

    colors = plt.cm.rainbow(np.linspace(0., 1., len(r_stats_list)))

    fig = plt.figure(figsize=(15,7))
    m = r_stats_list[0].max(); m=1
    plt.plot(x_var, r_stats_list[0]/m, color=colors[0], marker='o', label='RMS x {0:.3f}'.format(m))
    m = r_stats_list[1].max(); m=1
    plt.plot(x_var, r_stats_list[1]/m, color=colors[1], marker='o', label='Flux addition x {0:.3f}'.format(m))
    m = r_stats_list[2].max(); m=1
    plt.plot(x_var, r_stats_list[2]/m, color=colors[2], marker='o', label='Flux lost x {0:.3f}'.format(m))
    m = r_stats_list[3].max()
    #plt.plot(x_var, r_stats_list[3]/m, color=colors[3], marker='o', label='Sharpness x {0:.3f}'.format(m))
    plt.grid()
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.xlabel(x_label, fontsize=20)
    #plt.legend(bbox_to_anchor=(1.275, 1.0), prop={'size':15})
    plt.legend(loc='best', prop={'size':20})

    if name is not None: plt.savefig(name, format='eps', dpi=1000)
    plt.show()
