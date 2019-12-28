
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
LL


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

# Vectorized: correct!
#--------------------------
log_S = np.repeat(log(p), [N], axis=0).reshape(K,N).T + np.matmul(X, log(theta.T)) + np.matmul(1-X, log(1-theta.T))  

m = np.amax(log_S) 
s_n = np.sum(exp(log_S-m),axis=1)
#denom = np.repeat(1/s_n, [K], axis=0).reshape(N,K)
#Z_star = np.multiply(exp(log_S),denom)
LL_ = m + log(s_n).reshape(N,1)

LL_[:6]
LL[:6]

sum(LL)
sum(LL_)

def loglike_new(X, p, theta):
    
    """(Incomplete) Loglikelihood of Bernoulli mixture model"""
    
    N, K = X.shape[0], len(p)
    theta = theta.T       # Transpose for easier comparability with derivations
    S, LL = np.empty((N,K)), np.empty((N,1))

    log_S = np.repeat(log(p), [N], axis=0).reshape(K,N).T + np.matmul(X, log(theta.T)) + np.matmul(1-X, log(1-theta.T))      
    m = np.amax(log_S) 
    s_n = np.sum(exp(log_S-m),axis=1)
    LL_ = m + log(s_n).reshape(N,1)
    return sum(LL)

alphas = gamma(shape=1, size=K)               # shape parameters
p_0 = dirichlet(alpha = alphas, size = 1)[0]
theta_0 = beta(a = 1, b = 1, size = K*D).reshape(D,K)


loglike_new(X, p_0, theta_0)


loglike(X, p_0, theta_0)


