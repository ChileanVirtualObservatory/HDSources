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
        N = len(X)/4
        xc = X[0:N]
        yc = X[N:2*N]
        
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
        
        
        if self.square_c: c = X[2*N:3*N]**2
        else: c = X[2*N:3*N]
        sig = X[3*N:4*N]


        # computing the phi-matrices
        if self.compact_supp:
            phi_m = phi(Dx, Dy, sig.reshape(1,-1))
            if self.b!=0.:
                phix_m = phix(Dx, Dy, sig.reshape(1,-1))
                phiy_m = phiy(Dx, Dy, sig.reshape(1,-1))
                phixx_m = phixx(Dx, Dy, sig.reshape(1,-1))
                phixy_m = phixy(Dx, Dy, sig.reshape(1,-1))
                phiyy_m = phiyy(Dx, Dy, sig.reshape(1,-1))
        else:
            phi_m = phi(Dx, Dy, sig.reshape(1,-1), supp=0.)
            if self.b!=0.:
                phix_m = phix(Dx, Dy, sig.reshape(1,-1), supp=0.)
                phiy_m = phiy(Dx, Dy, sig.reshape(1,-1), supp=0.)
                phixx_m = phixx(Dx, Dy, sig.reshape(1,-1), supp=0.)
                phixy_m = phixy(Dx, Dy, sig.reshape(1,-1), supp=0.)
                phiyy_m = phiyy(Dx, Dy, sig.reshape(1,-1), supp=0.)

        
    
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
        
        
        # boundary conditions (threshold must be added)
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
            sol = sp.optimize.root(elf.F, np.concatenate([elf.xc, elf.yc, elf.c, elf.sig]), 
                                   method='lm', options={'maxiter':step_iter})
            sol_length = len(sol.x)/4
            opt_xc = sol.x[0:sol_length]
            opt_yc = sol.x[sol_length:2*sol_length]
            opt_c = sol.x[2*sol_length:3*sol_length]
            opt_sig = sol.x[3*sol_length:4*sol_length]
            
            # variation of c and sig
            delta_xc = np.linalg.norm(opt_xc-elf.xc)
            delta_yc = np.linalg.norm(opt_yc-elf.yc)
            delta_c = np.linalg.norm(opt_c-elf.c)
            delta_sig = np.linalg.norm(opt_sig-elf.sig)

            # searching for noisy gaussians (and removing them)
            mask = np.abs(opt_sig)<1.
            if np.any(~mask):
                print('Noisy gaussians detected and removed! \n')
                opt_c = opt_c[mask]
                opt_sig = opt_sig[mask]
                opt_xc = opt_xc[mask]
                opt_yc = opt_yc[mask]

            # update of best parameters
            elf.set_centers(opt_xc, opt_yc)
            elf.set_c(opt_c)
            elf.set_sig(opt_sig)
            
            # residual stats
            var,entr,rms = compute_residual_stats(elf.dfunc, opt_c, opt_sig, elf.xc, elf.yc, dims=elf.dims,
                           base_level=elf.base_level, square_c=elf.square_c, compact_supp=elf.compact_supp)
            
            # appending residual variance, entropy and rms
            residual_variance.append(var)
            residual_entropy.append(entr)
            residual_rms.append(rms)
            
            print('Variation on xc = {0}'.format(delta_xc))
            print('Variation on yc = {0}'.format(delta_yc))
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