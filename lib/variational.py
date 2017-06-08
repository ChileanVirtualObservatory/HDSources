import time
import scipy
import numba
import scipy as sp
import numpy as np
import numexpr as ne
from math import sqrt, exp
import matplotlib.pyplot as plt
import graph as gp
from utils import *
from hausdorff import hausdorff


#################################################################
# General Psi penalizing function (applicable in both cases)
#################################################################
def psi1(x, lamb=1.):
    x = lamb*x
    ret = np.empty(x.shape)
    mask0 = x<=0.
    mask1 = x>=1.
    mask01 = np.logical_and(np.logical_not(mask0),np.logical_not(mask1))
    ret[mask0] = 0.
    ret[mask1] = 1.
    #evaluation on 0-1
    x = x[mask01]
    ret[mask01] = ne.evaluate('10*x**3 - 15*x**4 + 6*x**5')
    return ret


def d1psi1(x, lamb=1.):
    x = lamb*x
    ret = np.empty(x.shape)
    mask0 = x<=0.
    mask1 = x>=1.
    mask01 = np.logical_and(np.logical_not(mask0),np.logical_not(mask1))
    ret[mask0] = 0.
    ret[mask1] = 0.
    #evaluation on 0-1
    x = x[mask01]
    ret[mask01] = ne.evaluate('30*x**2 - 60*x**3 + 30*x**4')
    return lamb*ret


def d2psi1(x, lamb=1.):
    x = lamb*x
    ret = np.empty(x.shape)
    mask0 = x<=0.
    mask1 = x>=1.
    mask01 = np.logical_and(np.logical_not(mask0),np.logical_not(mask1))
    ret[mask0] = 0.
    ret[mask1] = 0.
    #evaluation on 0-1
    x = x[mask01]
    ret[mask01] = ne.evaluate('60*x - 180*x**2 + 120*x**3')
    return (lamb**2)*ret


def psi2(x, lamb=1.):
    return lamb**2 * np.log(1 + x/lamb**2)


def d1psi2(x, lamb=1.):
    return 1./(1.+x/lamb**2)


def d2psi2(x, lamb=1.):
    return -lamb**2/(lamb**2+x)**2


#################################################################
# Euler-Lagrange class definition
#################################################################
class ELModel():
    def __init__(self, data, dfunc, dims, xe, ye, xc, yc, xb, yb, c0, sig0, d1psi1=d1psi1, 
                d1psi2=d1psi2, d2psi2=d2psi2, a=0., b=0., lamb1=1., lamb2=1., base_level=0.,
                minsig=None, maxsig=None, pix_freedom=1., support=5.):

        f0 = dfunc( np.vstack([xe,ye]).T )
        fb = dfunc( np.vstack([xb,yb]).T )
        len_f0 = len(f0)
        len_c0 = len(c0)
        len_sig0 = len(sig0)
        Ne = len(xe)
        Nc = len(xc)
        Nb = len(xb)

        
        # important atributes
        self.data = data
        if base_level > 0.:
            self.mask = data > base_level
        else: 
            self.mask = None
        self.dfunc = dfunc
        self.dims = dims
        self.f0 = f0
        self.fb = fb
        self.xb = xb; self.yb = yb
        self.xe = xe; self.ye = ye
        self.xc = xc; self.yc = yc
        self.xc0 = xc; self.yc0 = yc
        self.theta_xc = np.zeros(Nc); self.theta_yc = np.zeros(Nc)
        self.deltax = pix_freedom * 1./dims[0]
        self.deltay = pix_freedom * 1./dims[1]
        # minimal and maximum broadening
        if minsig is None:
            # reasoning: 3 minsig = pixsize / 2
            self.minsig = ( 0.5*(1./dims[0] + 1./dims[1]) ) / 6.
        else:
            self.minsig = minsig
        if maxsig is None:
            K = np.sum(self.mask)//Nc
            self.maxsig = K*self.minsig
        else:
            self.maxsig = maxsig
        # inverse transformation to (real) model parameters
        self.c = np.sqrt(c0)
        self.sig = inv_sig_mapping(sig0, minsig, maxsig)
        self.d1psi1 = d1psi1
        self.d1psi2 = d1psi2
        self.d2psi2 = d2psi2
        self.a = a
        self.b = b
        self.lamb1 = lamb1
        self.lamb2 = lamb2
        self.base_level = base_level
        self.support = support
        # solution variables
        self.scipy_sol = None
        self.elapsed_time = None
        self.residual_stats = None


    def set_centers(self, theta_xc, theta_yc):
        self.xc = self.xc0 + self.deltax * np.sin(theta_xc)
        self.yc = self.yc0 + self.deltay * np.sin(theta_yc)   


    def set_theta(self, theta_xc, theta_yc):
        self.theta_xc = theta_xc
        self.theta_yc = theta_yc


    def set_c(self, c):
        self.c = c


    def set_sig(self, sig):
        self.sig = sig


    def set_params(self, params):
        N = len(params)/4
        self.theta_xc = params[0:N]
        self.theta_yc = params[N:2*N]
        self.c = params[2*N:3*N]
        self.sig = params[3*N:4*N]


    def get_params(self):
        """
        Get the parameter of the function F (to optimize): 
        theta_xc, theta_yc, c, sig
        """
        return np.concatenate([self.theta_xc, self.theta_yc, self.c, self.sig])


    def get_params_mapped(self):
        """
        Get the real parameters of the model (mapped/bounded):
        xc, yc, c, sig
        """
        #xc = self.xc0 + self.deltax * np.sin(self.theta_xc)
        #yc = self.yc0 + self.deltay * np.sin(self.theta_yc)
        xc = self.xc
        yc = self.yc
        c = self.c**2
        sig = sig_mapping(self.sig, self.minsig, self.maxsig)
        return xc, yc, c, sig


    def get_w(self):
        """
        Get the mapping from the 'c' coefficients in the linear
        combination of Gausssian functions, to the 'w' in the
        linear combination of Normal functions. 
        """
        xc, yc, c, sig = self.get_params_mapped()
        d = len(self.dims)
        w = c * (2*np.pi*sig**2)**(d/2.)
        return w


    def get_residual_stats(self):
        _xe = np.linspace(0., 1., self.dims[0]+2)[1:-1]
        _ye = np.linspace(0., 1., self.dims[1]+2)[1:-1]
        Xe,Ye = np.meshgrid(_xe, _ye, sparse=False, indexing='ij')
        xe = Xe.ravel(); ye = Ye.ravel()

        xc, yc, c, sig = self.get_params_mapped()

        u = u_eval(c, sig, xc, yc, xe, ye, support=self.support) + self.base_level
        u = u.reshape(self.dims)

        if self.mask is not None:
            residual = self.data[self.mask]-u[self.mask]
        else:
            residual = self.data-u
        
        # first term of Lagrangian stats
        total_flux = np.sum(self.data[self.mask])
        flux_mask = residual<0.
        flux_addition = -1. * np.sum(residual[flux_mask])
        flux_lost = np.sum(residual[~flux_mask])
        psi1_int = np.sum(psi1(-1*residual))
        npix = np.sum(flux_mask)

        # second term of Lagrangian stats
        img_grad = gradient(u)
        sharpness = np.sum(img_grad)
        psi2_int = np.sum(psi2(img_grad))

        residual_stats = estimate_variance(residual), estimate_entropy(residual), \
                         estimate_rms(residual), flux_addition/total_flux, \
                         flux_lost/total_flux, psi1_int, npix, sharpness, psi2_int
        self.residual_stats = residual_stats

        return residual_stats


    def get_approximation(self):
        _xe = np.linspace(0., 1., self.dims[0]+2)[1:-1]
        _ye = np.linspace(0., 1., self.dims[1]+2)[1:-1]
        Xe,Ye = np.meshgrid(_xe, _ye, sparse=False, indexing='ij')
        xe = Xe.ravel(); ye = Ye.ravel()

        xc, yc, c, sig = self.get_params_mapped()

        u = u_eval(c, sig, xc, yc, xe, ye, support=self.support) + self.base_level
        u = u.reshape(self.dims)

        return u


    def get_gradient(self):
        _xe = np.linspace(0., 1., self.dims[0]+2)[1:-1]
        _ye = np.linspace(0., 1., self.dims[1]+2)[1:-1]
        Xe,Ye = np.meshgrid(_xe, _ye, sparse=False, indexing='ij')
        xe = Xe.ravel(); ye = Ye.ravel()

        xc, yc, c, sig = self.get_params_mapped()

        grad = grad_eval(c, sig, xc, yc, xe, ye, support=self.support) + self.base_level
        grad = grad.reshape(self.dims)

        return grad


    def prune(self):
        w = self.get_w()
        mask, _ = prune(w)
        #update all the arrays
        self.xc = self.xc[mask]; self.xc0 = self.xc0[mask]
        self.yc = self.yc[mask]; self.yc0 = self.yc0[mask]
        self.theta_xc = self.theta_xc[mask]
        self.theta_yc = self.theta_yc[mask]
        self.c = self.c[mask]
        self.sig = self.sig[mask]

    
    def summarize(self, solver_output=True, residual_stats=True, coverage_stats=True, homogeneity_stats=True,
                  solution_plot=True, params_plot=True, histograms_plot=True):
        print('\n \n' + '#'*90)    
        print('FINAL RESULTS:')
        print('#'*90 + '\n')
        _xc, _yc, _c, _sig = self.get_params_mapped()

        out = self.get_residual_stats()
        var,entr,rms,flux_addition,flux_lost,psi1_int,npix,sharpness,psi2_int = out
        
        if solver_output:
            print('Solver Output:')
            print('success: {0}'.format(self.scipy_sol['success']))
            print('status: {0}'.format(self.scipy_sol['status']))
            print('message: {0}'.format(self.scipy_sol['message']))
            print('nfev: {0}'.format(self.scipy_sol['nfev']))
        
        if residual_stats:
            print('\nResidual stats:')
            print('Residual RMS: {0}'.format(rms))
            print('Residual Variance: {0}'.format(var))
            #print('Residual Entropy: {0}'.format(entr))
            print('Flux Lost: {0}'.format(flux_lost))
            print('Flux Addition: {0}'.format(flux_addition))
            print('psi1(u-f): {0}'.format(psi1_int))
            print('Exceeded Pixels: {0}'.format(npix))
            print('Sharpness: {0}'.format(sharpness))
            print('psi2(grad u): {0}'.format(psi2_int))

            print('Total elapsed time: {0} [s]'.format(self.elapsed_time))

        if coverage_stats:
            center = np.ascontiguousarray( np.vstack( [self.xc, self.yc] ).T )
            colloc = np.ascontiguousarray( np.vstack( [self.xe, self.ye] ).T )
            print('\nCoverage of solution:')
            print('Hausdorff distance between collocation and center points: {0}'.format(hausdorff(colloc, center)))
            print('Mean min distance between collocation and center points: {0}'.format(mean_min_dist(colloc, center)))

        if homogeneity_stats:
            # we map each parameter vector to [0,1]
            __xc = _xc - _xc.min(); __xc /= __xc.max()
            __yc = _yc - _yc.min(); __yc /= __yc.max()
            __c = _c - _c.min(); __c /= __c.max()
            __sig = _sig - _sig.min(); __sig /= __sig.max()
            params = np.vstack([__xc, __yc, __c, __sig]).T
            params_dist_matrix = build_dist_matrix(params, inf=True)
            max_mdist = np.max( params_dist_matrix.min(axis=1) )
            mean_mdist = np.mean( params_dist_matrix.min(axis=1) )
            print('\nHomogeneity of solution:')
            print('Mean min distance in the (standarized) parameters space: {0}'.format(mean_mdist))
            print('Max min distance in the (standarized) parameters space: {0}'.format(max_mdist))

        
        if solution_plot:
            gp.solution_plot(self.dfunc, _c, _sig, _xc, _yc, dims=self.dims, 
                             base_level=self.base_level, support=self.support)
            #gp.image_plot(img_grad, title='Gradient')
            #gp.image_plot(img_sign, title='Exceeded Pixels')
        
        if params_plot:
            gp.params_plot(_c, _sig, _xc, _yc)
            gp.params_distribution_plot(_c, _sig)

        if histograms_plot:
            u = self.get_approximation()
            term1 = u[self.mask]-self.data[self.mask]
            plt.figure(figsize=(8,8))
            plt.hist(term1.ravel(), bins=10, facecolor='seagreen', edgecolor='black', lw=2)
            plt.title('u-f')
            plt.show()

            
    def F(self, params):
        N = len(params)//4
        theta_xc = params[0:N]
        theta_yc = params[N:2*N]

        # parameters transform/mapping
        xc = self.xc0 + self.deltax * np.sin(theta_xc)
        yc = self.yc0 + self.deltay * np.sin(theta_yc)
        c = params[2*N:3*N]**2
        sig = sig_mapping(params[3*N:4*N], self.minsig, self.maxsig)

        # if np.any(np.isnan(xc)):
        #     print("xc")
        # if np.any(np.isnan(yc)):
        #     print("yc")
        # if np.any(np.isnan(c)):
        #     print("c")
        # if np.any(np.isnan(sig)):
        #     print("sig")
        # if np.any(np.isnan(params)):
        #     return -1



        # evaluation points
        xe = np.hstack([self.xe, xc]); ye = np.hstack([self.ye, yc])
        #xe = self.xe; ye = self.ye
        xb = self.xb; yb = self.yb
        
        # computing u, ux, uy, ...
        if self.b==0.:
            u = u_eval(c, sig, xc, yc, xe, ye, support=self.support) + self.base_level
        else:
            out = u_eval_full(c, sig, xc, yc, xe, ye, support=self.support)
            u = out[0,:] + self.base_level
            ux = out[1,:]
            uy = out[2,:]
            uxy = out[3,:]
            uxx = out[4,:]
            uyy = out[5,:]
        
        # computing the EL equation
        f0 = np.hstack([ self.f0, self.dfunc(np.vstack([xc,yc]).T) ])
        #f0 = self.f0
        a = self.a; b = self.b; lamb1 = self.lamb1; lamb2 = self.lamb2

        tmp1 = ne.evaluate('u-f0')
        tmp2 = self.d1psi1(tmp1, lamb1)
        el = ne.evaluate('2*tmp1 + a*tmp2')

        if self.b!=0.:
            laplacian = ne.evaluate('ux**2 + uy**2')
            tmp1 = ne.evaluate('((ux*uxx + uy*uxy)*ux + (ux*uxy + uy+uyy)*uy)')
            tmp2 = ne.evaluate('uxx + uyy')
            tmp3 = self.d2psi2(laplacian, lamb2)
            tmp4 = self.d1psi2(laplacian, lamb2)
            el += ne.evaluate('2*b*(2* tmp3*tmp1  + tmp4*tmp2)')
            
        # evaluating at boundary
        fb = self.fb
        u_boundary = u_eval(c, sig, xc, yc, self.xb, self.yb, self.support) + self.base_level
        
        return np.concatenate([el,u_boundary-fb])


    def _F(self, params):
        N = len(params)//4
        xc = params[0:N]
        yc = params[N:2*N]

        c = params[2*N:3*N]
        sig = params[3*N:4*N]

        xe = self.xe; ye = self.ye
        xb = self.xb; yb = self.yb

        # computing u, ux, uy, ...
        u = u_eval(c, sig, xc, yc, xe, ye, support=self.support) + self.base_level
        
        # computing the EL equation
        f0 = self.f0; a = self.a; b = self.b; lamb1 = self.lamb1; lamb2 = self.lamb2

        tmp1 = ne.evaluate('u-f0')
        tmp2 = self.d1psi1(tmp1, lamb1)
        el = ne.evaluate('2*tmp1 + a*tmp2')

        if self.b!=0.:
            laplacian = ne.evaluate('ux**2 + uy**2')
            tmp1 = ne.evaluate('((ux*uxx + uy*uxy)*ux + (ux*uxy + uy+uyy)*uy)')
            tmp2 = ne.evaluate('uxx + uyy')
            tmp3 = self.d2psi2(laplacian, lamb2)
            tmp4 = self.d1psi2(laplacian, lamb2)
            el += ne.evaluate('2*b*(2* tmp3*tmp1  + tmp4*tmp2)')
        
        return el

    
#################################################################
# Euler-Lagrange instansiation solver
#################################################################

# NOTE: ADD VERBOSITY LEVEL

def elm_solver(elm, method='standard', max_nfev=None, n_iter=100, verbose=True, xtol=1.e-7, ftol=1.e-7):
    t0 = time.time()

    # if step_iter is None:
    #     step_iter = int(max_iter/10)

    # if method=='exact':
    #     residual_variance = []
    #     residual_entropy = []
    #     residual_rms = []
    #     iter_list = range(step_iter, max_iter+1, step_iter)
        
        # for it in iter_list:
        #     print('\n'+'#'*90)
        #     print('Results after {0} iterations'.format(it))
        #     print('#'*90)
        #     # lm optimization
        #     sol = sp.optimize.root(elm.F, elm.get_params(), method='lm', options={'maxiter':step_iter})
        #     sol_length = len(sol.x)/4
        #     opt_theta_xc = sol.x[0:sol_length]
        #     opt_theta_yc = sol.x[sol_length:2*sol_length]
        #     opt_c = sol.x[2*sol_length:3*sol_length]
        #     opt_sig = sol.x[3*sol_length:4*sol_length]

            #old_xc = elm.xc
            #old_yc = elm.yc
            #new_xc = elm.xc0 + elm.deltax * np.sin(opt_theta_xc)
            #new_yc = elm.yc0 + elm.deltay * np.sin(opt_theta_yc)
            
            # new_xc = elm.xc0 + elm.deltax * np.sin(np.pi * np.tanh(opt_theta_xc))
            # new_yc = elm.yc0 + elm.deltay * np.sin(np.pi * np.tanh(opt_theta_yc))
            
            # variation centers, c and sig
            #delta_theta_xc = np.linalg.norm(opt_theta_xc-elm.theta_xc) / np.linalg.norm(elm.theta_xc)
            #delta_theta_yc = np.linalg.norm(opt_theta_yc-elm.theta_yc) / np.linalg.norm(elm.theta_yc)
            #delta_xc = np.linalg.norm(new_xc-old_xc) / np.linalg.norm(old_xc)
            #delta_yc = np.linalg.norm(new_yc-old_yc) / np.linalg.norm(old_yc)
            #delta_c = np.linalg.norm(opt_c-elm.c) / np.linalg.norm(elm.c)
            #delta_sig = np.linalg.norm(opt_sig-elm.sig) / np.linalg.norm(elm.sig)

            # searching for noisy gaussians (and removing them)
            #mask = np.abs(opt_sig)<1.
            #if np.any(~mask):
            #    print('{0} noisy gaussians detected and removed! \n'.format(np.sum(~mask)))
            #    opt_theta_xc = opt_theta_xc[mask]
            #    opt_theta_yc = opt_theta_yc[mask]
            #    opt_c = opt_c[mask]
            #    opt_sig = opt_sig[mask]

            # update to best parameters
            # elm.set_theta(opt_theta_xc, opt_theta_yc)
            # elm.set_centers(opt_theta_xc, opt_theta_yc)
            # elm.set_c(opt_c)
            # elm.set_sig(opt_sig)
            
            # residual stats
            #var,entr,rms = elm.get_residual_stats()
            # var,entr,rms = compute_residual_stats(elm.data, elm.xc, elm.yc, elm.c, elm.sig, 
            #                                       dims=elm.dims, support=elm.support)
            
            # appending residual variance, entropy and rms
            #residual_variance.append(var)
            #residual_entropy.append(entr)
            #residual_rms.append(rms)

            #print('Variation on theta_xc = {0}'.format(delta_theta_xc))
            #print('Variation on theta_yc = {0}'.format(delta_theta_yc))
            #print('Variation on xc = {0}'.format(delta_xc))
            #print('Variation on yc = {0}'.format(delta_yc))
            #print('Variation on c = {0}'.format(delta_c))
            #print('variation on sig = {0}'.format(delta_sig))
            #print('\nsuccess: {0}'.format(sol['success']))
            #print('\nstatus: {0}'.format(sol['status']))
            #print('\nmessage: {0}'.format(sol['message']))
            #print('\nnfev: {0}'.format(sol['nfev']))
            #if sol['success']: break

    if method=='standard':
        # lm optimization from scipy.optimize.root
        options = {'maxiter':max_nfev, 'xtol':xtol, 'ftol':ftol}
        sol = sp.optimize.root(elm.F, elm.get_params(), method='lm', options=options)
        sol_length = len(sol.x)//4
        opt_theta_xc = sol.x[0:sol_length]
        opt_theta_yc = sol.x[sol_length:2*sol_length]
        opt_c = sol.x[2*sol_length:3*sol_length]
        opt_sig = sol.x[3*sol_length:4*sol_length]

        # update to best parameters
        elm.set_theta(opt_theta_xc, opt_theta_yc)
        elm.set_centers(opt_theta_xc, opt_theta_yc)
        elm.set_c(opt_c)
        elm.set_sig(opt_sig)

        # prune of gaussian in elm
        elm.prune()
    
    elif method=='iterative':
        for it in range(n_iter):
            print('\n'+'#'*90)
            print('Results after {0} iterations'.format(it+1))
            print('#'*90)
            
            # lm optimization from scipy.optimize.root
            options = {'maxiter':max_nfev, 'xtol':xtol, 'ftol':ftol}
            sol = sp.optimize.root(elm.F, elm.get_params(), method='lm', options=options)
            sol_length = len(sol.x)//4
            opt_theta_xc = sol.x[0:sol_length]
            opt_theta_yc = sol.x[sol_length:2*sol_length]
            opt_c = sol.x[2*sol_length:3*sol_length]
            opt_sig = sol.x[3*sol_length:4*sol_length]

            # update to best parameters
            elm.set_theta(opt_theta_xc, opt_theta_yc)
            elm.set_centers(opt_theta_xc, opt_theta_yc)
            elm.set_c(opt_c)
            elm.set_sig(opt_sig)
            
            print('\nsuccess: {0}'.format(sol['success']))
            print('\nstatus: {0}'.format(sol['status']))
            print('\nmessage: {0}'.format(sol['message']))
            print('\nnfev: {0}'.format(sol['nfev']))
            if sol['success']: break
                
    elm.scipy_sol = sol
    elm.elapsed_time = time.time() - t0
    elm.summarize()
    # print('Residual RMS: {0}'.format(residual_rms[-1]))
    # print('Residual Variance: {0}'.format(residual_variance[-1]))
    # print('Residual Entropy: {0}'.format(residual_entropy[-1]))
    # print('Total elapsed time: {0} [s]'.format(time.time()-t0))
    
    # plots generation
    #residual_plot(residual_variance, residual_entropy, residual_rms, iter_list[0:len(residual_rms)])
