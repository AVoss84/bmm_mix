
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
denom = np.repeat(1/s_n, [K], axis=0).reshape(N,K)
Z_star = np.multiply(exp(log_S),denom)
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
        
        if u[k,:] == 0.: 
            u[k,:]  = 10**(-10)
            
        theta_new[k,:] = v[k,]/u[k,:] 
    p_new = u/N ;   
    assert round(sum(p_new),2) == 1., 'Step 6 does not produce probabilities!'
    return p_new, theta_new.T

Z_star = Z
p_basic, th_basic = M_step(X, Z_star)
p_basic

u, v, theta_new = np.empty((K,1)), np.empty((K,D)), np.empty((K,D))
for k in range(K):
    for d in range(D):
        v[k,d] = sum(Z_star[:,k]*X[:,d])
    u[k,:] = sum(Z_star[:,k])
    
    if u[k,:] == 0.: 
        u[k,:]  = 10**(-10)
        
    theta_new[k,:] = v[k,]/u[k,:] 
v
u    

p_new = u/N ;   
p_new
assert round(sum(p_new),2) == 1., 'Step 6 does not produce probabilities!'


v_new = np.matmul(Z_star.T, X)
v_new
v_new.shape
v

u_new = np.sum(Z_star,axis=0).reshape(K,1)
u_new
u
u.shape
u_new.shape

theta_new[1,:]
v[1,]/u[1,:] 

denom = np.repeat(u_new, [D], axis=0).reshape(K,D)
denom.shape
denom

denom = np.where(denom==0, 10**(-10), denom)

theta_New = np.multiply(v_new, 1/denom)
theta_New
theta_New.shape

theta_new
theta_new.shape


def M_step_vec(X, Z_star):
    """
    Maximization step: steps 5-7 in Algorithm 1
    """
    N, D, K = X.shape[0], X.shape[1], Z_star.shape[1]
    #u, v, theta_new = np.empty((K,1)), np.empty((K,D)), np.empty((K,D))
    v = np.matmul(Z_star.T, X)
    u = np.sum(Z_star,axis=0)#.reshape(K,1)
    denom = np.repeat(u, [D], axis=0).reshape(K,D)
    denom = np.where(denom==0, 10**(-10), denom)    
    theta_new = np.multiply(v, 1/denom)
    p_new = u/N ;   
    assert round(sum(p_new),2) == 1., 'Step 6 does not produce probabilities!'
    return p_new, theta_new.T

M_step_vec(X, Z_star)
M_step(X, Z_star)
