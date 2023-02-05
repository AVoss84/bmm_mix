
import numpy as np
from numpy import log, sum, exp, prod
from numpy.random import beta, binomial, dirichlet, multinomial, uniform, gamma, seed, standard_gamma, gumbel
from copy import deepcopy
import pandas as pd
#from tqdm.auto import tqdm
from tqdm import tqdm
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#from scipy.stats import wishart, multivariate_normal, bernoulli, multinomial
#import os
#from scipy.stats import multinomial

# Draw from D-variate Bernoulli mixture:
#-----------------------------------------
def sample_bmm(N, p, theta):
    """"
    Draw from D-variate Bernoulli mixture
    N: random sample size
    p: vector of mixture weights, prior mixture weights
    theta: matrix of success prob. of each dimension, per mixture comp.
    """    
    d = theta.shape[0]
    Z = multinomial(1, p, size=N)       # draw from categorical p.m.f
    print("Sampling from", d,"dimensional Bernoulli mixture with", len(p), "mixture components.")
    print("Number of random draws:", N)
    # Draw X_i| Z:  
    X = np.empty((N,d))
    for i in tqdm(range(N)):
        select_comp = np.where(Z[i,:])[0][0]         # select mixture comp.
        X[i,:] = binomial(n=1, p = theta[:,select_comp])  # Select success prob. of each d=1..D dimension in mix. comp. k. Draw from d-variate Bernoulli draw
    return X, Z
#--------------------------------------------------------------------------------   
    

#---------------------------------------------------------------------------------
def E_step_basic(X, theta, p, jitter=10**(-5)):
    """
    Expectation step: see steps 3-4 of Algorithm 1
    """
    N = X.shape[0]; K = len(p)
    theta = theta.T                # Transpose for easier comparability with derivations
    S, Z_star, LL = np.empty((N,K)), np.empty((N,K)), np.empty((N,1))
    #noise = 0. #uniform(0,jitter,1)
    for n in range(N):
       for k in range(K):
            log_s_nk = log(p[k]) + sum(X[n,:]*log(theta[k,:]) + (1-X[n,:])*log(1-theta[k,:]))
            #s_nk = p[k]*prod(( theta[k,:]**(X[n,:]) ) * ( (1 - (theta[k,:]) )**(1-X[n,:]) ))          
            #s_nk = exp(log(p[k]+noise) + sum(X[n,:]*log(theta[k,:]+noise) + (1-X[n,:])*log(1 - (theta[k,:]+noise) )))  # step 3
            #S[n,k] = s_nk
            S[n,k] = exp(log_s_nk)     # y_nk
       
       Z_star[n,:] = S[n,:]/sum(S[n,:])     # step 4; posterior of mixture assignments
       LL[n,:] = log(sum(S[n,:]))
       
       #marg = sum(S[n,:])     
       #if marg == 0.:
       #   marg = 10**(-5) 
       #   print("zero sum.")
       #Z_star[n,:] = S[n,:]/marg     # step 4; posterior of mixture assignments
       #LL[n,:] = log(marg)
       #m = np.amax(S[n,:]) 
       #LL[n,:] = m + log(sum(exp(S[n,:]-m)))
       #denom = exp(LL[n,:])
       #if denom == 0.:
       #     denom = 10**(-10)
       #Z_star[n,:] = exp(S[n,:])/denom
       #s_n = exp(S[n,:]-m)
       #Z_star[n,:] = S_n]/sum(S[n,:])     # step 4; posterior of mixture assignments
    return sum(LL), Z_star

#------------------------------------------------------------------------------------------------------

def E_step(X, theta, p):    
    """Expectation step: see steps 3-4 of Algorithm 1"""  
    N = X.shape[0]; K = len(p)
    theta = theta.T                # Transpose for easier comparability with derivations
    
    # Vectorized version
    #--------------------------    
    log_S = np.repeat(log(p), [N], axis=0).reshape(K,N).T + np.matmul(X, log(theta.T + 1e-10)) + np.matmul(1-X, log(1-theta.T))  
    log_S = np.nan_to_num(log_S)
    
    b = np.max(log_S)    # exp-normalize-trick
    s_n = np.sum(exp(log_S - b),axis=1)    
    denom = np.repeat(1/s_n, [K], axis=0).reshape(N,K)    
    Z_star = np.multiply(exp(log_S - b),denom)

    #s_n = np.sum(exp(log_S),axis=1)    
    #denom = np.repeat(1/s_n, [K], axis=0).reshape(N,K)    
    #Z_star = np.multiply(exp(log_S),denom)
    #LL = log(s_n).reshape(N,1)
    
    LL = b + log(s_n).reshape(N,1)
    return sum(LL), Z_star

#------------------------------------------------------------------------------
    
# without exp-normalize trick...
def E_step_old(X, theta, p):
    
    """Expectation step: see steps 3-4 of Algorithm 1"""
    
    N = X.shape[0]; K = len(p)
    theta = theta.T                # Transpose for easier comparability with derivations
    
    # Vectorized version
    #--------------------------
    log_S = np.repeat(log(p), [N], axis=0).reshape(K,N).T + np.matmul(X, log(theta.T)) + np.matmul(1-X, log(1-theta.T))  
    log_S = np.nan_to_num(log_S)
    
    #b = np.max(log_S)    # exp-normalize-trick
    #s_n = np.sum(exp(log_S - b),axis=1)    
    #denom = np.repeat(1/s_n, [K], axis=0).reshape(N,K)    
    #Z_star = np.multiply(exp(log_S - b),denom)

    s_n = np.sum(exp(log_S),axis=1)    
    denom = np.repeat(1/s_n, [K], axis=0).reshape(N,K)    
    Z_star = np.multiply(exp(log_S),denom)
    
    LL = log(s_n).reshape(N,1)
    return sum(LL), Z_star
#------------------------------------------------------------------------------

def M_step(X, Z_star):
    """
    Maximization step: steps 5-7 in Algorithm 1
    """
    N, D, K = X.shape[0], X.shape[1], Z_star.shape[1]
    v = np.matmul(Z_star.T, X)
    u = np.sum(Z_star,axis=0)#.reshape(K,1)
    denom = np.repeat(u, [D], axis=0).reshape(K,D)
    denom = np.where(denom==0, 10**(-10), denom)    
    theta_new = np.multiply(v, 1/denom)
    p_new = u/sum(u) ;   
    assert round(sum(p_new),2) == 1., 'Step 6 does not produce probabilities!'
    return p_new, theta_new.T

#-----------------------------------------------------------------------------------
def M_step_basic(X, Z_star):
    """
    Maximization step: steps 5-7 in Algorithm 1
    """
    N, D, K = X.shape[0], X.shape[1], Z_star.shape[1]
    u, v, theta_new = np.empty((K,1)), np.empty((K,D)), np.empty((K,D))
    for k in range(K):
        for d in range(D):
            v[k,d] = sum(Z_star[:,k]*X[:,d])
        u[k,:] = sum(Z_star[:,k])
        
        if u[k,:] == 0.: 
            u[k,:]  = 10**(-10)
            
        theta_new[k,:] = v[k,]/u[k,:] 
    p_new = u/N ;   
    assert round(sum(p_new),2) == 1., 'Step 6 does not produce probabilities!'
    return p_new, theta_new.T
#-------------------------------------------------------------------------
 
def loglike(X, p, theta):
    """(Incomplete) Loglikelihood of Bernoulli mixture model"""
    N, K = X.shape[0], len(p)
    theta = theta.T       # Transpose for easier comparability with derivations
    log_S = np.repeat(log(p), [N], axis=0).reshape(K,N).T + np.matmul(X, log(theta.T)) + np.matmul(1-X, log(1-theta.T))      
    log_S = np.nan_to_num(log_S)
    
    m = np.amax(log_S) 
    s_n = np.sum(exp(log_S-m),axis=1)
    LL = m + log(s_n).reshape(N,1)
    return sum(LL)


#--------------------------------------------------------------------------
def loglike_basic(X, p, theta):
    """(Incomplete) Loglikelihood of Bernoulli mixture model"""
    N, K = X.shape[0], len(p)
    theta = theta.T       # Transpose for easier comparability with derivations
    S, LL = np.empty((N,K)), np.empty((N,1))
    for n in range(N):
       for k in range(K):
           
            try:
               y_nk = log(p[k]) + sum(X[n,:]*log(theta[k,:]) + (1-X[n,:])*log(1-theta[k,:]))
            except :
               print("Exception in loglikelihood computation!") 
               pass 
            #s_nk = log(p[k]) + sum(X[n,:]*log(theta[k,:]) + (1-X[n,:])*log(1-theta[k,:]))            
            #s_nk = p[k]*prod(( theta[k,:]**(X[n,:]) ) * ( (1 - (theta[k,:]) )**(1-X[n,:]) ))            
            #S[n,k] = s_nk
            S[n,k] = y_nk
       #b = np.amax(S)     
       #LL[n,:] = log(sum(S[n,:])) 
       m = np.amax(S[n,:]) 
       LL[n,:] = m + log(sum(exp(S[n,:]-m)))
       #LL[n,:] = log(marg)
    return sum(LL)
#------------------------------------------------------------------------   

def mixture_EM(X: np.array, p_0: np.array, theta_0: np.array, n_iter: int=100, stopcrit: int=10**(-5), verbose : bool = True): 
    """Fit Benoulli mixture model via EM algorithm.  
    Args:
        X (np.array): _description_
        p_0 (np.array): _description_
        theta_0 (np.array): _description_
        n_iter (int, optional): _description_. Defaults to 100.
        stopcrit (int, optional): _description_. Defaults to 10**(-5).
        verbose (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    p_current, theta_current = p_0, theta_0
    ll = [loglike(X, p_0, theta_0)]        # store log-likehoods for each iteration
    i, converged, local, delta_ll = 0, 0, 0, 10**5 ;
    
    while i < n_iter:
    #for i in tqdm(range(n_iter), total=n_iter):    
        # E-step and :
        log_likes_t1, Z_star_new = E_step(X, theta_current, p_current)
        # M-step
        p_update, theta_update = M_step(X, Z_star_new)        
        # Incomplete loglikelihood:
        log_likes_t = loglike(X, p_update, theta_update)         
        #if np.isnan(log_likes_t):
        #   log_likes_t = log_likes_t1        
        ll.append(log_likes_t)
        delta_ll_old = delta_ll
        delta_ll = log_likes_t - log_likes_t1
        if verbose and (i % 20 == 0): print(i,"- delta LL.:", delta_ll)
        converged += (abs(delta_ll) < stopcrit)*1.
        local += (abs(delta_ll) > abs(delta_ll_old))*1
        
        if converged >= 5:
          print("Stop criterion applied!")
          print(delta_ll)
          break
        elif i == (n_iter-1):
          print("Convergence not guaranteed.")            
        elif local >= 30:
          print("Local optimum reached.")    
          print(delta_ll)
          break
        else:
          p_current, theta_current = p_update, theta_update  
        i += 1
    return ll, p_update, theta_update, Z_star_new.argmax(axis=1)     # use argmax of last iteration of Zstar's as assignements estimates
#------------------------------------------------------------------
    
def dirichlet_sample(alphas):
    """
    Generate samples from an array of alpha distributions.
    """
    r = standard_gamma(alphas)
    return r / r.sum(-1, keepdims=True)


def discrete_sample(alphas):
    """
    Draw from categorical distr. using the Gumbel-max-trick.
    https://casmls.github.io/general/2017/02/01/GumbelSoftmax.html
    """
    N,K = alphas.shape
    #uniform = np.random.rand(len(alpha))     # uniform random draws
    #gumbels = -log(-log(uniform)) + log(alphas)
    gumbels = log(alphas) + gumbel(loc=0, scale=1, size=alphas.shape)
    categ_level = gumbels.argmax(axis=1)
    cat_array = np.repeat(categ_level, [K], axis=0).reshape(N,K)
    seq = np.arange(0,K,1)
    seq_array = np.repeat(seq, [N], axis=0).reshape(K,N).T
    onehot = np.equal(seq_array, cat_array)*1.
    return categ_level, onehot

#----------------------------------------   
# Single Gibbs draw for all unknowns:
#----------------------------------------    
def gibbs_pass(p_old, thetas_old, X, alphas = np.array([.1,.3,.6]),
               hyper_para = {'gammas': np.array([.1,.3,.6]), 'deltas': np.array([1,3,6])}
               ):
    """Single Gibbs iteration for all unknowns variables, i.e. latents and model parameters"""
    
    p = deepcopy(p_old)
    theta = deepcopy(thetas_old)     
    N, D, K = X.shape[0], X.shape[1], len(p)
    
    gammas = np.repeat(hyper_para['gammas'], [D], axis=0).reshape(K,D)
    deltas = np.repeat(hyper_para['deltas'], [D], axis=0).reshape(K,D)
    theta = theta.T                # Transpose for easier comparability with derivations
    
    # Vectorized version
    #--------------------------
    log_S = np.repeat(log(p), [N], axis=0).reshape(K,N).T + np.matmul(X, log(theta.T)) + np.matmul(1-X, log(1-theta.T))  
    log_S = np.nan_to_num(log_S)
    
    b = np.max(log_S)    # exp-normalize-trick
    #b = 0.
    s_n = np.sum(exp(log_S - b),axis=1)    
    denom = np.repeat(1/s_n, [K], axis=0).reshape(N,K)    
    S_n = np.multiply(exp(log_S - b),denom)
            
    cat_lev, Z_star = discrete_sample(S_n)  # draw from categorical p.m.f
    
    v = np.matmul(Z_star.T, X)
    u = np.sum(Z_star,axis=0)
    us = np.repeat(u, [D], axis=0).reshape(K,D)
    
    p_new = dirichlet_sample(alphas + u)
    thetas_new = beta(a = gammas + v, b = deltas + us - v, size = v.shape)    
    return cat_lev, p_new, thetas_new.T

