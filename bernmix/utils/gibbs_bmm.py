
# single GIbbs cycle:

p = p_0
N = X.shape[0]; K = len(p)
theta = theta_0.T                # Transpose for easier comparability with derivations
S, Z_star, LL = np.empty((N,K)), np.empty((N,K)), np.empty((N,1))

X.shape
theta.shape
S.shape

K
n = 1
k = 1

log(p[k]) + sum(X[n,:]*log(theta[k,:]) + (1-X[n,:])*log(1-theta[k,:]))

np.repeat(log(p), [N], axis=0).reshape(K,N).T + np.matmul(X, log(theta.T)) + np.matmul(1-X, log(1-theta.T))#.shape



for n in range(N):
   for k in range(K):
       
        log_s_nk = log(p[k]) + sum(X[n,:]*log(theta[k,:]) + (1-X[n,:])*log(1-theta[k,:]))
        #s_nk = p[k]*prod(( theta[k,:]**(X[n,:]) ) * ( (1 - (theta[k,:]) )**(1-X[n,:]) ))
        
        #S[n,k] = s_nk
        S[n,k] = exp(log_s_nk)  #exp(log_s_nk)     # y_nk

        #S[n,k] = log_s_nk  #exp(log_s_nk)     # y_nk
   s_n = sum(S[n,:])     
   S_n = S[n,:]/s_n  
   Z_star[n,:] = S_n   
   #Z_star[n,:] = multinomial(1, S_n, size=1)       # draw from categorical p.m.f
   LL[n,:] = log(s_n)



# Vectorized: correct!
#--------------------------
log_S = np.repeat(log(p), [N], axis=0).reshape(K,N).T + np.matmul(X, log(theta.T)) + np.matmul(1-X, log(1-theta.T))#.shape
S.shape

exp(log_S)

s_n = np.sum(exp(log_S),axis=1)
s_n

log(s_n)

den = np.repeat(1/s_n, [K], axis=0).reshape(N,K)#.T

np.multiply(exp(log_S),den)
Z_star

LL[0:10]

LL = log(s_n).reshape(N,1)



def Evec_step(X, theta, p, jitter=10**(-5)):
    
    """Expectation step: see steps 3-4 of Algorithm 1"""
    
    N = X.shape[0]; K = len(p)
    theta = theta.T                # Transpose for easier comparability with derivations
    S, Z_star, LL = np.empty((N,K)), np.empty((N,K)), np.empty((N,1))
    
    # Vectorized: correct!
    #--------------------------
    log_S = np.repeat(log(p), [N], axis=0).reshape(K,N).T + np.matmul(X, log(theta.T)) + np.matmul(1-X, log(1-theta.T))  
    s_n = np.sum(exp(log_S),axis=1)
    denom = np.repeat(1/s_n, [K], axis=0).reshape(N,K)
    Z_star = np.multiply(exp(log_S),denom)
    LL = log(s_n).reshape(N,1)
    return sum(LL), Z_star

E_step(X, theta = theta_0, p = p_0)

Evec_step(X, theta = theta_0, p = p_0)





