import sys
import numpy as np
import scipy as sp
import sympy as sym
import numexpr as ne
import matplotlib.pyplot as plt
from scipy import interpolate, optimize


"""
HELPER FUNCTIONS TO VISUALIZE AND COMPARE SOLUTIONS
"""
def compare_plot(c, sig, xc, resolution=10, title=None, log=False):
    _xe = np.linspace(0., 1., 10*N, endpoint=True)
    _Dx = np.empty((10*N,N))
    for k in range(10*N):
        _Dx[k,:] = (_xe[k] - xc)

    phi_m = phi(_Dx, sig)
    u = np.dot(phi_m, c**2)
    plt.figure(figsize=(10,6))
    plt.plot(_xe, u, 'r-', label='Solution')
    plt.plot(x_, f(x_), 'b--', label='Data')
    plt.plot(xe, f(xe), 'go', label='Evaluation points')
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.3, 1.0))
    plt.show()

def compare_plot_log(c, sig, xc, resolution=10, title=None):
    _xe = np.linspace(0., 1., 10*N, endpoint=True)
    _Dx = np.empty((10*N,N))
    for k in range(10*N):
        _Dx[k,:] = (_xe[k] - xc)

    phi_m = phi(_Dx, sig)
    u = np.dot(phi_m, c**2)
    plt.figure(figsize=(10,6))
    plt.semilogy(_xe, np.abs(u-f(_xe)), 'r-')
    plt.title(title)
    plt.show()



"""
PSI1 PENALIZING FUNCTIONS AND ITS DERIVATIVES
"""
def d1psi1(x, lamb=1.):
    x = lamb*x
    ret = np.empty(x.shape)
    mask0 = x<=0.
    mask1 = x>=1.
    mask01 = np.logical_and(np.logical_not(mask0),np.logical_not(mask1))
    ret[mask0] = 0.
    ret[mask1] = 1.
    #evaluation on 0-1
    x = x[mask01]
    ret[mask01] = ne.evaluate('(4*(x-1)*x+2) / (4*(x-1)**2 * x**2 * (cosh(1/x + 1/(x-1))+1))')
    return lamb*ret



    
"""
PSI2 PENALIZING FUNCTIONS AND ITS DERIVATIVES
"""
#Perona-Malik regularizer
def psi2(x, lamb=1.):
    return lamb**2 * np.log(1. + x/lamb**2)

def d1psi2(x, lamb=1.):
    return 1. / (1. + x/lamb**2)

def d2psi2(x, lamb=1.):
    return -lamb**2 / (x + lamb**2)**2



"""
GENERAL PSI PENALIZING FUNCTION (APPLICABLE IN BOTH CASES)
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
RBF (GAUSSIAN) FUNCTION AND ITS DERIVATIVES
"""
def phi(x, sig, minsig=0):
    retval = ne.evaluate('exp(-x**2/(2*(minsig**2+sig**2)))')
    return retval

def phix(x, sig, minsig=0):
    retval = ne.evaluate('-(1./(minsig**2+sig**2)) * exp(-x**2/(2*(minsig**2+sig**2))) * x')
    return retval

def phixx(x, sig, minsig=0):
    retval = ne.evaluate('(1./(minsig**2+sig**2)**2) * exp(-x**2/(2*(minsig**2+sig**2))) * (x**2 - minsig**2 - sig**2)')
    return retval



"""
CLASS TO HANDLE VARIATIONAL PROBLEM WITH (C,SIG) VARIATING TOGETHER 
"""
class ELFunc1():
    def __init__(self, f, xe, xc, c0, sig0, d1psi1=None, d2psi2=None, a=0., b=0., lamb1=1., lamb2=1.):
        #data function at evaluation and boundary points
        f0 = f(xe)
        fb = [f(0),f(1)]
        
        len_f0 = len(f0)
        len_xe = len(xe)
        len_xc = len(xc); N = len_xc;
        len_c0 = len(c0)
        len_sig0 = len(sig0)
        """
        TODO:verify consistency
        """
        #if len_c0 != len_sig0: 
        #    print('Dimensions of c0 and sig0 must match!')
        #    return None
        #if (shape_Dx[0]+2 != shape_Dx[1]) or (shape_Dx[1] != len_c0):
        #    print('Dimensions of Dx are wrong!')
        #    return None
        """ 
        Computing distance matrix.
        Note: Evaluation and collocation points will be the same
        """
        Dx = np.empty((2*N-2,N))
        for i in range(2*N-2):
            Dx[i,:] = (xe[i] - xc)
        
        self.f0 = f0
        self.fb = fb
        self.xe = xe
        self.xc = xc
        self.Dx = Dx
        self.c = c0
        self.sig = sig0
        self.phi_m   = phi(Dx,sig0)
        self.phix_m  = phix(Dx,sig0)
        self.phixx_m = phixx(Dx,sig0)
        self.d1psi1 = d1psi1
        self.d2psi2 = d2psi2
        self.a = a
        self.b = b
        self.lamb1 = lamb1
        self.lamb2 = lamb2
    
    def set_c(self, c):
        self.c = c
    
    def set_sig(self, sig):
        self.sig = sig
        self.phi_m   = phi(self.Dx,sig)
        self.phix_m  = phix(self.Dx,sig)
        self.phixx_m = phixx(self.Dx,sig)
    
    """
    Function to optimize c and sig together
    """
    def F(self, X):
        #unpacking parameters
        N = len(X)/2
        c_squared = X[0:N]**2
        sig = X[N:]

        #phi function's evaluation
        phi_m   = phi(self.Dx, sig)
        phix_m  = phix(self.Dx, sig)
        phixx_m = phixx(self.Dx, sig)
        

        #computing the Euler-Lagrange equation
        u   = np.dot(phi_m, c_squared)
        ux  = np.dot(phix_m, c_squared)
        uxx = np.dot(phixx_m, c_squared)
        el = 2.*(u-self.f0) + \
            self.a*self.d1psi1(u-self.f0, lamb=self.lamb1) - \
            self.b*uxx*self.d2psi2(np.abs(ux), lamb=self.lamb2)
        
        #evaluating at boundary
        bc = [np.dot(phi(-self.xc,sig),c_squared)-self.fb[0], np.dot(phi(1.-self.xc,sig),c_squared)-self.fb[-1]]
        return np.concatenate([el,bc])



"""
CLASS TO HANDLE VARIATIONAL PROBLEM WITH (C,SIG) VARIATING SEPARATELY
"""
class ELFunc2():
    def __init__(self, f, xe, xc, c0, sig0, d1psi1=None, d2psi2=None, a=0., b=0., lamb1=1., lamb2=1.):
        #data function at evaluation and boundary points
        f0 = f(xe)
        fb = [f(0),f(1)]
        
        len_f0 = len(f0)
        len_xe = len(xe)
        len_xc = len(xc); N = len_xc;
        len_c0 = len(c0)
        len_sig0 = len(sig0)
        """
        TODO:verify consistency
        """
        #if len_c0 != len_sig0: 
        #    print('Dimensions of c0 and sig0 must match!')
        #    return None
        #if (shape_Dx[0]+2 != shape_Dx[1]) or (shape_Dx[1] != len_c0):
        #    print('Dimensions of Dx are wrong!')
        #    return None
        """ 
        Computing distance matrix.
        Note: Evaluation and collocation points will be the same
        """
        Dx = np.empty((2*N-2,N))
        for i in range(2*N-2):
            Dx[i,:] = (xe[i] - xc)
        
        self.f0 = f0
        self.fb = fb
        self.xe = xe
        self.xc = xc
        self.Dx = Dx
        self.c = c0
        self.sig = sig0
        self.phi_m   = phi(Dx,sig0)
        self.phix_m  = phix(Dx,sig0)
        self.phixx_m = phixx(Dx,sig0)
        self.d1psi1 = d1psi1
        self.d2psi2 = d2psi2
        self.a = a
        self.b = b
        self.lamb1 = lamb1
        self.lamb2 = lamb2
    
    def set_c(self, c):
        self.c = c
    
    def set_sig(self, sig):
        self.sig = sig
        self.phi_m   = phi(self.Dx,sig)
        self.phix_m  = phix(self.Dx,sig)
        self.phixx_m = phixx(self.Dx,sig)
        
    """
    Function to optimize c parameters
    """
    def F1(self, c):
        c_squared = c**2
        #computing the Euler-Lagrange equation
        u   = np.dot(self.phi_m, c_squared)
        ux  = np.dot(self.phix_m, c_squared)
        uxx = np.dot(self.phixx_m, c_squared)
        el  = 2.*(u-self.f0) + \
            self.a*self.d1psi1(u-self.f0, lamb=self.lamb1) - \
            self.b*uxx*self.d2psi2(np.abs(ux), lamb=self.lamb2)

        #evaluating at boundary
        bc = [np.dot(phi(-self.xc,self.sig),c_squared)-self.fb[0], 
              np.dot(phi(1.-self.xc,self.sig),c_squared)-self.fb[-1]]
        return np.concatenate([el,bc])

    """
    Function to optimize sig parameters
    """
    def F2(self, sig):    
        #phi function's evaluation
        phi_m   = phi(self.Dx, sig)
        phix_m  = phix(self.Dx, sig)
        phixx_m = phixx(self.Dx, sig)
        
        #c squared
        c_squared = self.c**2
        
        #computing the Euler-Lagrange equation
        u   = np.dot(phi_m, c_squared)
        ux  = np.dot(phix_m, c_squared)
        uxx = np.dot(phixx_m, c_squared)
        el = 2.*(u-self.f0) + \
            self.a*self.d1psi1(u-self.f0, self.lamb1) - \
            self.b*uxx*self.d2psi2(np.abs(ux), self.lamb2)

        #evaluating at boundary
        bc = [np.dot(phi(-self.xc,sig),c_squared) - self.fb[0], 
              np.dot(phi(1.-self.xc,sig),c_squared) - self.fb[-1]]
        return np.concatenate([el,bc])




"""
SOLVE FUNCTIONS TO HANDLE ELFunc2 INSTANCES
"""
def solve(elf, n_iter=3, verbose=True):
    for i in range(n_iter):
        #solve for c
        sol = sp.optimize.root(elf.F1, elf.c, method='lm', options={'maxiter':10000})
        opt_c = sol.x
        delta_c = np.linalg.norm(opt_c-elf.c)
        elf.set_c(opt_c)
        title = 'Best solution at iter={0} and improved c'.format(i)
        compare_plot(elf.c, elf.sig, elf.xc, title=title)
        print('Variation on c={0}'.format(delta_c))
        print('\nnfev: {0}'.format(sol['nfev']))
        print('\nmessage: {0}'.format(sol['message']))
        print('\nsuccess: {0}'.format(sol['success']))
        print('------------------------------------------------------------')

        #solve for sig
        sol = sp.optimize.root(elf.F2, elf.sig, method='lm', options={'maxiter':10000})
        opt_sig = sol.x
        delta_sig = np.linalg.norm(opt_sig-elf.sig)
        elf.set_sig(opt_sig)
        title = 'Best solution at iter={0} and improved sig'.format(i)
        compare_plot(elf.c, elf.sig, elf.xc, title=title)
        print('Variation on sig={0}'.format(delta_sig))
        print('\nnfev: {0}'.format(sol['nfev']))
        print('\nmessage: {0}'.format(sol['message']))
        print('\nsuccess: {0}'.format(sol['success']))
        print('-------------------------------------------------------------')
