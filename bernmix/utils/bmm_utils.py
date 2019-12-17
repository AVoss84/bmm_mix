
import numpy as np
from numpy import log, sum, exp, prod
from numpy.random import beta, binomial, dirichlet, multinomial, uniform, gamma, seed


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
    Z = multinomial(1, p, size=N)
    print("Sampling from", d,"dimensional Bernoulli mixture with", len(p), "mixture components.")
    print("Number of random draws:", N)
    # Draw X_i| Z:  
    X = np.empty((N,d))
    for i in range(N):
        select_comp = np.where(Z[i,:])[0][0]         # select mixture comp.
        X[i,:] = binomial(n=1, p = theta[:,select_comp])  # Select success prob. of each d=1..D dimension in mix. comp. k. Draw from d-variate Bernoulli draw
    return X, Z
#--------------------------------------------------------------------------------   
    

#---------------------------------------------------------------------------------
def E_step(X, theta, p, jitter=10**(-5)):
    
    """Expectation step: see steps 3-4 of Algorithm 1"""
    
    N = X.shape[0]; K = len(p)
    theta = theta.T                # Transpose for easier comparability with derivations
    S, Z_star, LL = np.empty((N,K)), np.empty((N,K)), np.empty((N,1))
    #noise = 0. #uniform(0,jitter,1)
    for n in range(N):
       for k in range(K):
            s_nk = p[k]*prod(( theta[k,:]**(X[n,:]) ) * ( (1 - (theta[k,:]) )**(1-X[n,:]) ))
            #s_nk = exp(log(p[k]+noise) + sum(X[n,:]*log(theta[k,:]+noise) + (1-X[n,:])*log(1 - (theta[k,:]+noise) )))  # step 3
            S[n,k] = s_nk
       #Z_star[n,:] = S[n,:]/sum(S[n,:])     # step 4; posterior of mixture assignments
       #LL[n,:] = log(sum(S[n,:]))
       marg = sum(S[n,:])     
       if marg == 0.:
          marg = 10**(-5) 
          print("zero sum.")
       Z_star[n,:] = S[n,:]/marg     # step 4; posterior of mixture assignments
       LL[n,:] = log(marg)
    return sum(LL), Z_star

#------------------------------------------------------------------------------
    
def M_step(X, Z_star):
    """
    Maximization step: steps 5-7 in Algorithm 1
    """
    N, D, K = X.shape[0], X.shape[1], Z_star.shape[1]
    u, v, theta_new = np.empty((K,1)), np.empty((K,D)), np.empty((K,D))
    for k in range(K):
        for d in range(D):
            v[k,d] = sum(Z_star[:,k]*X[:,d])
        u[k,:] = sum(Z_star[:,k])
        theta_new[k,:] = v[k,]/u[k,:] 
    p_new = u/N ;   
    assert round(sum(p_new),2) == 1., 'Step 6 does not produce probabilities!'
    return p_new, theta_new.T
#-------------------------------------------------------------------------
 

#------------------------------------------------------------------------
def loglike(X, p, theta):
    
    """(Incomplete) Loglikelihood of Bernoulli mixture model"""
    
    N, K = X.shape[0], len(p)
    theta = theta.T       # Transpose for easier comparability with derivations
    S, LL = np.empty((N,K)), np.empty((N,1))
    for n in range(N):
       for k in range(K):
            s_nk = p[k]*prod(( theta[k,:]**(X[n,:]) ) * ( (1 - (theta[k,:]) )**(1-X[n,:]) ))
            #s_nk = exp(log(p_k[k]) + sum(X[n,:]*log(theta[k,:]) + (1-X[n,:])*log(1-theta[k,:])))  # step 3
            S[n,k] = s_nk
       #LL[n,:] = log(sum(S[n,:])) 
       marg = sum(S[n,:])     
       if marg == 0.:
          marg = 10**(-5) 
          print("zero sum.")
       LL[n,:] = log(marg)
    return sum(LL)
#------------------------------------------------------------------------   


def mixture_EM(X, p_0, theta_0, n_iter=100, stopcrit=10**(-5)): 
    
    p_current, theta_current = p_0, theta_0
    ll = [loglike(X, p_0, theta_0)]        # store log-likehoods for each iteration
    i, converged = 0, 0 ;
    
    while i < n_iter :
        # E-step and loglikelihood for current parameters:
        log_likes_t1, Z_star_new = E_step(X, theta_current, p_current)
        
        p_update, theta_update = M_step(X, Z_star_new)        
        
        log_likes_t = loglike(X, p_update, theta_update)    
        
        ll.append(log_likes_t)
        delta_ll = log_likes_t - log_likes_t1
        print(i,"- delta LL.:", delta_ll)
        converged += (abs(delta_ll) < stopcrit)*1.
        
        if converged >= 2:
          print("Stop criterion applied!\n")
          print(delta_ll)
          break;
        else:
          p_current, theta_current = p_update, theta_update             
        i += 1
    return ll, p_update, theta_update
#------------------------------------------------------------------

