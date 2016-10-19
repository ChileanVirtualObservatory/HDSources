import matplotlib.pyplot as plt
import numpy as np



def plotter(c, sig, xc, resolution=10, title=None):
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
    plt.plot(x_, f(x_), 'b--', label='Data')
    plt.plot(xe, f(xe), 'go', label='Evaluation points')
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.3, 1.0))
    plt.show()

def plot_sol(c, sig, xc, yc, base_level=0., square_c=True, resolution=5, title=None, compact_supp=True):
    _xe = np.linspace(0., 1., 41*resolution)[1:-1]
    _ye = np.linspace(0., 1., 41*resolution)[1:-1]
    len_xe = len(_xe); len_ye = len(_ye)
    Xe,Ye = np.meshgrid(_xe, _ye, sparse=False)
    xe = Xe.ravel(); ye = Ye.ravel()
    Nc = len(xc)
    Ne = len(xe)
    
    """ 
    Computing distance matrices
    """
    #distance matrices
    Dx = np.empty((Ne,Nc))
    Dy = np.empty((Ne,Nc))
    for k in range(Ne):
        Dx[k,:] = xe[k]-xc
        Dy[k,:] = ye[k]-yc
    """
    Computing the Phi matrix
    """
    if square_c: c = c**2
    if compact_supp: phi_m = phi(Dx, Dy, sig.reshape(1,-1))
    else: phi_m = phi(Dx, Dy, sig.reshape(1,-1), supp=0.)
    u = np.dot(phi_m, c) + base_level
    u = u.reshape(len_xe, len_ye)

    """
    2D plot
    """
    plt.figure(figsize=(18,12))
    plt.subplot(1,3,1)
    ax = plt.gca()
    im = ax.imshow(f(_xe, _ye))
    #plt.imshow(np.log10(np.abs(u)+1e-10))
    plt.title('Original')
    plt.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    
    """
    2D plot
    """
    plt.subplot(1,3,2)
    ax = plt.gca()
    im = ax.imshow(u)
    #plt.imshow(np.log10(np.abs(u)+1e-10))
    plt.title('Solution')
    plt.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    
    """
    2D plot
    """
    plt.subplot(1,3,3)
    ax = plt.gca()
    im = ax.imshow(f(_xe, _ye)-u)
    #plt.imshow(np.log10(np.abs(u)+1e-10))
    plt.title('Residual')
    plt.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.show()
    
    #X,Y = np.meshgrid(_xe, _ye,sparse=True)
    #fig = plt.figure(figsize=(15,8))
    #ax = fig.add_subplot(111, projection='3d')
    #ax.plot_surface(X, Y, u, linewidth=0.1, cmap='jet')
    #ax.set_xlabel('x')
    #ax.set_ylabel('y')
    #ax.set_zlabel('I')
    #plt.title('Solution')
    #plt.show()
    residual = f(_xe, _ye)-u
    return (estimate_variance(residual), 
            estimate_entropy(residual),
            estimate_rms(residual))

def params_plot(c, sig, xc, yc, square_c=True):
    if square_c: c = c**2
    plt.figure(figsize=(17,7))
    plt.subplot(1,2,1)
    plt.title('Plot of c parameters')
    plt.xlim(-0.01,1.01)
    plt.ylim(-0.01,1.01)
    plt.scatter(xc, yc, c=c)
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.subplot(1,2,2)
    plt.title('Plot of sig^2 parameters')
    plt.xlim(-0.01,1.01)
    plt.ylim(-0.01,1.01)
    plt.scatter(xc, yc, c=sig**2)
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.show()
    
def params_distribution_plot(c, sig, square_c=True, percentile=100.):
    if square_c: c = c**2
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