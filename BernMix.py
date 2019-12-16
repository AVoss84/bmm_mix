
import numpy as np
from numpy import log, sum, exp, prod
from numpy.random import beta, binomial, dirichlet, uniform, gamma, seed, multinomial
#from scipy.stats import multinomial
import bmm_utils as bmm
from imp import reload
import matplotlib.pyplot as plt

reload(bmm)

seed(12)

K = 3           # number of mixture components
D = 10           # dimensions / number of features     

alphas = gamma(shape=1, size=K)               # shape parameter
p_true = dirichlet(alpha = alphas, size = 1)[0]
#p_true = np.array([.3,.6,.1])  # K>2
p_true

#K = len(p_k)
theta_true = beta(a = .7, b = .9, size = K*D).reshape(D,K)
theta_true

#mu_ks = mu_k[np.argmax(Z, axis=1)]

# Sample from mixture model
#-----------------------------
# Draw from Bernoulli:
#probs = np.random.uniform(size=10000)
#rbern = (np.random.uniform(size=D) < mu_k[:,Z[2]]) * 1
X, Z = bmm.sample_bmm(500, p_true, theta_true)

X.shape
Z.shape

# Set starting values for parameters:
#----------------------------------------
#seed(12)

K = 3           # number of mixture components
D = X.shape[1]

#alphas = gamma(shape=1, size=K)               # shape parameter
#p_k = dirichlet(alpha = alphas, size = 1)[0]
p_0 = np.array([1/K]*K)  # K>2
theta_0 = beta(a = 1, b = 1, size = K*D).reshape(D,K)

# Run EM:    
#----------
logli, p_em, theta_em = bmm.mixture_EM(X, p_0, theta_0, n_iter=100, stopcrit=10**(-4))

# Plot loglike.:
#----------------
plt.plot(logli, 'b--')
plt.title("Convergence check")
plt.xlabel('iterations')
plt.ylabel('loglikelihood')
plt.show()

p_em
p_true

theta_em
theta_true

# https://www.python-course.eu/neural_network_mnist.php

image_size = 28 # width and length
no_of_different_labels = 10 #  i.e. 0, 1, 2, 3, ..., 9
image_pixels = image_size * image_size

data_path = "~/data/mnist/"

train_data = np.loadtxt(data_path + "mnist_train.csv", 
                        delimiter=",")
test_data = np.loadtxt(data_path + "mnist_test.csv", 
                       delimiter=",") 
test_data[:10]

test_data[test_data==255]
test_data.shape

fac = 0.99 / 255
train_imgs = np.asfarray(train_data[:, 1:]) * fac + 0.01
test_imgs = np.asfarray(test_data[:, 1:]) * fac + 0.01

train_labels = np.asfarray(train_data[:, :1])
test_labels = np.asfarray(test_data[:, :1])


