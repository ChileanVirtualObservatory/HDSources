import scipy
import scipy as sp
import numpy as np
import numexpr as ne
import matplotlib.pyplot as plt
from graph import  solution_plot, params_plot, params_distribution_plot, residual_plot
from utils import *


"""
General Psi penalizing function (applicable in both cases)
"""
def psi(x, lamb=1.):
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

def d1psi(x, lamb=1.):
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

def d2psi(x, lamb=1.):
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


"""
Euler-Lagrange class definition
"""
class ELFunc():
    def __init__(self, dfunc, dims, xe, ye, xc, yc, xb, yb, c0, sig0, d1psi1=None, d1psi2=None, d2psi2=None,
                 a=0., b=0., lamb1=1., lamb2=1., base_level=0, square_c=True, compact_supp=False):
        f0 = np.array([dfunc(xe[i],ye[i]) for i in range(len(xe))]).ravel()
        fb = np.array([dfunc(xb[i],yb[i]) for i in range(len(xb))]).ravel()
        len_f0 = len(f0)
        len_c0 = len(c0)
        len_sig0 = len(sig0)
        Ne = len(xe)
        Nc = len(xc)
        Nb = len(xb)

        """ 
        Computing distance matrices
        """
        #distance matrices
        Dx = np.empty((Ne,Nc))
        Dy = np.empty((Ne,Nc))
        for k in range(Ne):
            Dx[k,:] = xe[k]-xc
            Dy[k,:] = ye[k]-yc
        #distance matrices for boundary points
        Dxb = np.empty((Nb,Nc))
        Dyb = np.empty((Nb,Nc))
        for k in range(Nb):
            Dxb[k,:] = xb[k]-xc
            Dyb[k,:] = yb[k]-yc
            
        """
        Computing Phi matrices
        """
        if compact_supp:
            phi_m = phi(Dx, Dy, sig0.reshape(1,-1))
            if b!=0.:
                phix_m = phix(Dx, Dy, sig0.reshape(1,-1))
                phiy_m = phiy(Dx, Dy, sig0.reshape(1,-1))
                phixx_m = phixx(Dx, Dy, sig0.reshape(1,-1))
                phixy_m = phixy(Dx, Dy, sig0.reshape(1,-1))
                phiyy_m = phiyy(Dx, Dy, sig0.reshape(1,-1))
        else:
            phi_m = phi(Dx, Dy, sig0.reshape(1,-1), supp=0.)
            if b!=0.:
                phix_m = phix(Dx, Dy, sig0.reshape(1,-1), supp=0.)
                phiy_m = phiy(Dx, Dy, sig0.reshape(1,-1), supp=0.)
                phixx_m = phixx(Dx, Dy, sig0.reshape(1,-1), supp=0.)
                phixy_m = phixy(Dx, Dy, sig0.reshape(1,-1), supp=0.)
                phiyy_m = phiyy(Dx, Dy, sig0.reshape(1,-1), supp=0.)

        
        """
        Storing important atributes
        """
        self.dfunc = dfunc
        self.dims = dims
        self.f0 = f0
        self.fb = fb
        self.xe = xe; self.ye = ye
        self.xc = xc; self.yc = yc
        self.xb = xb; self.yb = yb
        self.Dx = Dx; self.Dxb = Dxb
        self.Dy = Dy; self.Dyb = Dyb
        self.phi_m = phi_m
        if b!=0.:
            self.phix_m = phix_m
            self.phiy_m = phiy_m
            self.phixx_m = phixx_m
            self.phiyy_m = phiyy_m
            self.phixy_m = phixy_m
        self.c = c0
        self.sig = sig0
        self.d1psi1 = d1psi1
        self.d1psi2 = d1psi2
        self.d2psi2 = d2psi2
        self.a = a
        self.b = b
        self.lamb1 = lamb1
        self.lamb2 = lamb2
        self.base_level = base_level
        self.square_c = square_c
        self.compact_supp = compact_supp


    def set_centers(self, xc, yc):
        xe = self.xe; ye = self.ye
        xb = self.xb; yb = self.yb
        Ne = len(xe)
        Nc = len(xc)
        Nb = len(xb)

        # re-computing distance matrices
        Dx = np.empty((Ne,Nc))
        Dy = np.empty((Ne,Nc))
        for k in range(Ne):
            Dx[k,:] = xe[k]-xc
            Dy[k,:] = ye[k]-yc

        # re-computing distance matrices for boundary points
        Dxb = np.empty((Nb,Nc))
        Dyb = np.empty((Nb,Nc))
        for k in range(Nb):
            Dxb[k,:] = xb[k]-xc
            Dyb[k,:] = yb[k]-yc
        
        self.xc = xc; self.yc = yc
        self.Dx = Dx; self.Dy = Dy
        self.Dxb = Dxb; self.Dyb = Dyb

        
    def set_c(self, c):
        self.c = c

    def set_sig(self, sig):
        self.sig = sig
        """
        Re-computing Phi matrices
        """
        if self.compact_supp:
            self.phi_m = phi(self.Dx, self.Dy, sig.reshape(1,-1))
            if self.b!=0.:
                self.phix_m = phix(self.Dx, self.Dy, sig.reshape(1,-1))
                self.phiy_m = phiy(self.Dx, self.Dy, sig.reshape(1,-1))
                self.phixx_m = phixx(self.Dx, self.Dy, sig.reshape(1,-1))
                self.phixy_m = phixy(self.Dx, self.Dy, sig.reshape(1,-1))
                self.phiyy_m = phiyy(self.Dx, self.Dy, sig.reshape(1,-1))
        else:
            self.phi_m = phi(self.Dx, self.Dy, sig.reshape(1,-1), supp=0.)
            if self.b!=0.:
                self.phix_m = phix(self.Dx, self.Dy, sig.reshape(1,-1), supp=0.)
                self.phiy_m = phiy(self.Dx, self.Dy, sig.reshape(1,-1), supp=0.)
                self.phixx_m = phixx(self.Dx, self.Dy, sig.reshape(1,-1), supp=0.)
                self.phixy_m = phixy(self.Dx, self.Dy, sig.reshape(1,-1), supp=0.)
                self.phiyy_m = phiyy(self.Dx, self.Dy, sig.reshape(1,-1), supp=0.)

    
    def F(self, X):
        N = len(X)/2
        if self.square_c: c = X[0:N]**2
        else: c = X[0:N]
        sig = X[N:]
        
        """
        Computing the Phi-matrices
        """
        if self.compact_supp:
            phi_m = phi(self.Dx, self.Dy, sig.reshape(1,-1))
            if self.b!=0.:
                phix_m = phix(self.Dx, self.Dy, sig.reshape(1,-1))
                phiy_m = phiy(self.Dx, self.Dy, sig.reshape(1,-1))
                phixx_m = phixx(self.Dx, self.Dy, sig.reshape(1,-1))
                phixy_m = phixy(self.Dx, self.Dy, sig.reshape(1,-1))
                phiyy_m = phiyy(self.Dx, self.Dy, sig.reshape(1,-1))
        else:
            phi_m = phi(self.Dx, self.Dy, sig.reshape(1,-1), supp=0.)
            if self.b!=0.:
                phix_m = phix(self.Dx, self.Dy, sig.reshape(1,-1), supp=0.)
                phiy_m = phiy(self.Dx, self.Dy, sig.reshape(1,-1), supp=0.)
                phixx_m = phixx(self.Dx, self.Dy, sig.reshape(1,-1), supp=0.)
                phixy_m = phixy(self.Dx, self.Dy, sig.reshape(1,-1), supp=0.)
                phiyy_m = phiyy(self.Dx, self.Dy, sig.reshape(1,-1), supp=0.)

        
    
        # computing u, ux, uy, ...
        u = np.dot(phi_m, c) + self.base_level
        if self.b!=0.:
            ux = np.dot(phix_m, c)
            uy = np.dot(phiy_m, c)
            uxx = np.dot(phixx_m, c)
            uyy = np.dot(phiyy_m, c)
            uxy = np.dot(phixy_m, c)
        
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

        
        """
        Boundary conditions (threshold must be added)
        """
        bc = np.dot(phi(self.Dxb, self.Dyb, sig.reshape(1,-1)), c) + self.base_level - self.fb
        return np.concatenate([el,bc])
    
    def F1(self, c):
        if self.square_c: c = c**2

        """
        Computing u, ux, uy, ...
        """
        u = np.dot(phi_m, c) + self.base_level
        if self.b!=0.:
            ux = np.dot(phix_m, c)
            uy = np.dot(phiy_m, c)
            uxx = np.dot(phixx_m, c)
            uyy = np.dot(phiyy_m, c)
            uxy = np.dot(phixy_m, c)


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
        
        """
        Boundary conditions
        """
        bc = np.dot(phi(self.Dxb, self.Dyb, self.sig.reshape(1,-1)), c) + self.base_level - self.fb
        return np.concatenate([el,bc])
        
    def F2(self, sig):
        if self.square_c: c = self.c**2
        else: c = self.c

        """
        Computing the Phi-matrices
        """
        if self.compact_supp:
            phi_m = phi(self.Dx, self.Dy, sig.reshape(1,-1))
            if self.b!=0.:
                phix_m = phix(self.Dx, self.Dy, sig.reshape(1,-1))
                phiy_m = phiy(self.Dx, self.Dy, sig.reshape(1,-1))
                phixx_m = phixx(self.Dx, self.Dy, sig.reshape(1,-1))
                phixy_m = phixy(self.Dx, self.Dy, sig.reshape(1,-1))
                phiyy_m = phiyy(self.Dx, self.Dy, sig.reshape(1,-1))
        else:
            phi_m = phi(self.Dx, self.Dy, sig.reshape(1,-1), supp=0.)
            if self.b!=0.:
                phix_m = phix(self.Dx, self.Dy, sig.reshape(1,-1), supp=0.)
                phiy_m = phiy(self.Dx, self.Dy, sig.reshape(1,-1), supp=0.)
                phixx_m = phixx(self.Dx, self.Dy, sig.reshape(1,-1), supp=0.)
                phixy_m = phixy(self.Dx, self.Dy, sig.reshape(1,-1), supp=0.)
                phiyy_m = phiyy(self.Dx, self.Dy, sig.reshape(1,-1), supp=0.)
        

        # computing u, ux, uy, ...
        u = np.dot(phi_m, c) + self.base_level
        if self.b!=0.:
            ux = np.dot(phix_m, c)
            uy = np.dot(phiy_m, c)
            uxx = np.dot(phixx_m, c)
            uyy = np.dot(phiyy_m, c)
            uxy = np.dot(phixy_m, c)
        

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
        
        """
        Boundary conditions
        """
        bc = np.dot(phi(self.Dxb, self.Dyb, sig.reshape(1,-1)), c) + self.base_level - self.fb
        return np.concatenate([el,bc])


"""
Euler-Lagrange instansiation solver
"""

### ADD VERBOSE LEVEL

def el_solver(elf, method='exact', n_iter=None, step_iter=1000, max_iter=100000, mask=None, verbose=True):
    # number of centers/parameters
    Nc = len(elf.xc)

    if method=='exact':
        residual_variance = []
        residual_entropy = []
        residual_rms = []
        iter_list = range(step_iter, max_iter+1, step_iter)
        
        for it in iter_list:
            print('\n'+'#'*90)
            print('Results after {0} iterations'.format(it))
            print('#'*90)
            # lm optimization
            sol = sp.optimize.root(elf.F, np.concatenate([elf.c, elf.sig]), method='lm', options={'maxiter':step_iter})
            sol_length = len(sol.x)
            opt_c = sol.x[0:sol_length/2]
            opt_sig = sol.x[sol_length/2:]
            
            # variation of c and sig
            delta_c = np.linalg.norm(opt_c-elf.c)
            delta_sig = np.linalg.norm(opt_sig-elf.sig)

            # searching for noisy gaussians (and removing them)
            mask = np.abs(opt_sig)<1.
            if np.any(~mask):
                print('Noisy gaussians detected and removed! \n')
                opt_c = opt_c[mask]
                opt_sig = opt_sig[mask]
                xc = elf.xc[mask]
                yc = elf.yc[mask]
                elf.set_centers(xc, yc)

            # update of best parameters
            elf.set_c(opt_c)
            elf.set_sig(opt_sig)
            
            # residual stats
            var,entr,rms = compute_residual_stats(elf.dfunc, opt_c, opt_sig, elf.xc, elf.yc, dims=elf.dims,
                           base_level=elf.base_level, square_c=elf.square_c, compact_supp=elf.compact_supp)
            
            # appending residual variance, entropy and rms
            residual_variance.append(var)
            residual_entropy.append(entr)
            residual_rms.append(rms)
            
            print('Variation on c = {0}'.format(delta_c))
            print('variation on sig = {0}'.format(delta_sig))
            print('\nsuccess: {0}'.format(sol['success']))
            print('\nstatus: {0}'.format(sol['status']))
            print('\nmessage: {0}'.format(sol['message']))
            print('\nnfev: {0}'.format(sol['nfev']))
            if sol['success']: break
        
        print('\n \n' + '#'*90)    
        print('FINAL RESULTS:')
        print('#'*90)
        
        # plots generation
        solution_plot(elf.dfunc, opt_c, opt_sig, elf.xc, elf.yc, dims=elf.dims, base_level=elf.base_level, 
                      mask=mask, square_c=elf.square_c, compact_supp=elf.compact_supp)
        params_plot(elf.c, elf.sig, elf.xc, elf.yc, square_c=elf.square_c)
        params_distribution_plot(elf.c, elf.sig, square_c=elf.square_c)
        residual_plot(residual_variance, residual_entropy, residual_rms, iter_list[0:len(residual_rms)])
    
    if method=='mixed':
        print('\n'+'#'*90)

        print('Iteration: 0  -  Optimization on both c and sig parameters')
        print('#'*90)
        sol = sp.optimize.root(elf.F, np.concatenate([elf.c, elf.sig]), method='lm', options={'maxiter':10000}, callback=calllback)
        opt_c = sol.x[0:Nc]
        opt_sig = sol.x[Nc:]
        delta_c = np.linalg.norm(opt_c-elf.c)
        delta_sig = np.linalg.norm(opt_sig-elf.sig)
        elf.set_c(opt_c)
        elf.set_sig(opt_sig)
        solution_plot(elf.dfunc, elf.c, elf.sig, elf.xc, elf.yc, dims=elf.dims, base_level=elf.base_level, 
                            mask=mask, square_c=elf.square_c, compact_supp=elf.compact_supp)
        params_plot(elf.c, elf.sig, elf.xc, elf.yc, square_c=elf.square_c)
        params_distribution_plot(elf.c, elf.sig, square_c=elf.square_c)

        print('Variation on c={0}'.format(delta_c))
        print('Variation on sig={0}'.format(delta_sig))
        print('\nmessage: {0}'.format(sol['message']))
        print('\nsuccess: {0}'.format(sol['success']))
    
    if method=='iterative' or method=='mixed':
        residual_variance = []
        residual_entropy = []
        residual_rms = []
        
        #print('\n'+'#'*90)
        #print('Initial Guess')
        #print('#'*90)
        solution_plot(elf.dfunc, elf.c, elf.sig, elf.xc, elf.yc, dims=elf.dims, base_level=elf.base_level,
                      mask=mask, square_c=elf.square_c)
        #params_plot(elf.c, elf.sig, elf.xc, elf.yc, square_c=elf.square_c)
        #params_distribution_plot(elf.c, elf.sig, square_c=elf.square_c)
        #residual_variance.append(var)
        #residual_entropy.append(entr)
        #residual_rms.append(rms)
        
        for it in range(n_iter):
            print('\n'+'#'*90)
            print('Iteration: {0}  -  Optimization on c parameter'.format(it))
            print('#'*90)
            #solve for c
            sol = sp.optimize.root(elf.F1, elf.c, method='lm', options={'maxiter':10000})
            opt_c = sol.x
            delta_c = np.linalg.norm(opt_c-elf.c)
            elf.set_c(opt_c)
            #title = 'Best solution at iter={0} and improved c'.format(i)
            solution_plot(elf.dfunc, elf.c, elf.sig, elf.xc, elf.yc, dims=elf.dims, base_level=elf.base_level, 
                          mask=mask, square_c=elf.square_c, compact_supp=elf.compact_supp)
            params_plot(elf.c, elf.sig, elf.xc, elf.yc, square_c=elf.square_c)
            params_distribution_plot(elf.c, elf.sig, square_c=elf.square_c)
            print('Variation on c={0}'.format(delta_c))
            print('\nnfev: {0}'.format(sol['nfev']))
            print('\nmessage: {0}'.format(sol['message']))
            print('\nsuccess: {0}'.format(sol['success']))
            if elf.square_c:
                print('\nmax c and position: {0} and {1}'.format(np.max(opt_c**2), np.argmax(opt_c**2)))
            else:
                print('\nmax c and position: {0} and {1}'.format(np.max(opt_c), np.argmax(opt_c)))

            # residual stats
            var,entr,rms = compute_residual_stats(opt_c, opt_sig, elf.xc, elf.yc, dims=elf.dims, base_level=elf.base_level, 
                                                  square_c=elf.square_c, compact_supp=elf.compact_supp)
            
            #appending residual variance and entropy
            residual_variance.append(var)
            residual_entropy.append(entr)
            residual_rms.append(rms)

            print('\n'+'#'*90)
            print('Iteration: {0}  -  Optimization on sig parameter'.format(it))
            print('#'*90)
            #solve for sig
            sol = sp.optimize.root(elf.F2, elf.sig, method='lm', options={'maxiter':10000})
            opt_sig = sol.x
            delta_sig = np.linalg.norm(opt_sig-elf.sig)
            elf.set_sig(opt_sig)
            #title = 'Best solution at iter={0} and improved sig'.format(i)
            solution_plot(elf.dfunc, elf.c, elf.sig, elf.xc, elf.yc, dims=elf.dims, base_level=elf.base_level, 
                          mask=mask, square_c=elf.square_c, compact_supp=elf.compact_supp)
            params_plot(elf.c, elf.sig, elf.xc, elf.yc, square_c=elf.square_c)
            params_distribution_plot(elf.c, elf.sig, square_c=elf.square_c)
            
            print('Variation on sig={0}'.format(delta_sig))
            print('\nnfev: {0}'.format(sol['nfev']))
            print('\nmessage: {0}'.format(sol['message']))
            print('\nsuccess: {0}'.format(sol['success']))
            print('\nmax sig and position: {0} and {1}'.format(np.max(opt_sig**2), np.argmax(opt_sig**2)))
            print('\nmin sig and position: {0} and {1}'.format(np.min(opt_sig**2), np.argmin(opt_sig**2)))
            print('-------------------------------------------------------------')
            
            # residual stats
            var,entr,rms = compute_residual_stats(opt_c, opt_sig, elf.xc, elf.yc, dims=elf.dims, base_level=elf.base_level, 
                                                  square_c=elf.square_c, compact_supp=elf.compact_supp)

            #appending residual variance, entropy and RMS
            residual_variance.append(var)
            residual_entropy.append(entr)
            residual_rms.append(rms)

        print('\n \n' + '#'*90)    
        print('SOME FINAL RESULTS:')
        print('#'*90)
        plt.figure(figsize=(12,5))
        plt.subplot(1,3,1)
        plt.xlim(-0.2, (2.*n_iter-1)+0.2)
        plt.plot(range(2*n_iter), residual_rms, marker='o', c='g', s=.1)
        plt.title('Residual RMS')        
        plt.subplot(1,3,2)
        plt.xlim(-0.2, (2.*n_iter-1)+0.2)
        plt.plot(range(2*n_iter), residual_variance, marker='o', c='b', s=.1)
        plt.title('Residual variance')
        plt.subplot(1,3,3)
        plt.xlim(-0.2, (2.*n_iter-1)+0.2)
        plt.plot(range(2*n_iter), residual_entropy, marker='o', c='b', s=.1)
        plt.title('Residual entropy')
        plt.show()
