
import os, pickle, copy
import numpy as np
from numpy import log, sum, exp, prod
from numpy.random import beta, binomial, dirichlet, uniform, gamma, seed, multinomial
#from scipy.stats import multinomial
from imp import reload
import matplotlib.pyplot as plt
from bernmix.utils import bmm_utils as bmm

import bernmix

reload(bmm)

seed(12)

K = 3           # number of mixture components
D = 10           # dimensions / number of features     

#alphas = gamma(shape=1, size=K)               # shape parameter
#p_true = dirichlet(alpha = alphas, size = 1)[0]
p_true = np.array([.3,.6,.1])  # K>2
p_true

theta_true = beta(a = .7, b = .9, size = K*D).reshape(D,K)


# Sample from mixture model
#-----------------------------
# Draw from Bernoulli:
#probs = np.random.uniform(size=10000)
#rbern = (np.random.uniform(size=D) < mu_k[:,Z[2]]) * 1
X, Z = bmm.sample_bmm(1000, p_true, theta_true)

X.shape
Z.shape

# Set starting values for parameters:
#----------------------------------------
#seed(12)

K = 3           # number of mixture components
D = X.shape[1]

#alphas = gamma(shape=1, size=K)               # shape parameters
#p_0 = dirichlet(alpha = alphas, size = 1)[0]
p_0 = np.array([1/K]*K)  # K>2
theta_0 = beta(a = 1, b = 1, size = K*D).reshape(D,K)

#----------
# Run EM:    
#----------
logli, p_em, theta_em = bmm.mixture_EM(X = X, p_0 = p_0, theta_0 = theta_0, n_iter = 100, stopcrit = 10**(-3))


# Plot loglike.:
#----------------
plt.plot(logli, 'b--')
plt.title("Convergence check")
plt.xlabel('iterations')
plt.ylabel('loglikelihood')
plt.show()

# Compare:
p_em
p_true

theta_em
theta_true


################# REAL DATA ############################
########################################################

image_size = 28 # width and length
no_of_different_labels = 10 
image_pixels = image_size**2
image_pixels

# Read data in:
#os.getcwd()
data_path = "C:\\Users\\Alexander\\Documents\\Github\\BMM\\data\\mnist\\"

train_data = np.loadtxt(data_path + "mnist_train.csv", 
                        delimiter=",")

test_data = np.loadtxt(data_path + "mnist_test.csv", 
                       delimiter=",") 

test_data[:10]

test_data[test_data==255]
test_data.shape

train_imgs = np.asfarray(train_data[:, 1:])/255  # we avoid 0 values as inputs
test_imgs = np.asfarray(test_data[:, 1:])/255

X = train_imgs.copy()

#fac = 0.99 / 255
#train_imgs = np.asfarray(train_data[:, 1:]) * fac + 0.01  # we avoid 0 values as inputs which are capable of preventing weight updates
#test_imgs = np.asfarray(test_data[:, 1:]) * fac + 0.01

train_labels = np.asfarray(train_data[:, :1])
test_labels = np.asfarray(test_data[:, :1])


lr = np.arange(10)

for label in range(10):
    one_hot = (lr==label).astype(np.int)
    print("label: ", label, " in one-hot representation: ", one_hot)


lr = np.arange(no_of_different_labels)

# transform labels into one hot representation
train_labels_one_hot = (lr==train_labels).astype(np.float)
test_labels_one_hot = (lr==test_labels).astype(np.float)

# we don't want zeroes and ones in the labels neither:
train_labels_one_hot[train_labels_one_hot==0] = 0.01
train_labels_one_hot[train_labels_one_hot==1] = 0.99
test_labels_one_hot[test_labels_one_hot==0] = 0.01
test_labels_one_hot[test_labels_one_hot==1] = 0.99


#------ Plot images -------------------------------
for i in range(10):
    img = train_imgs[i].reshape((28,28))
    plt.imshow(img, cmap="Greys")
    plt.show()
#------------------------------------------------------


# Save images for later:
with open(data_path+"mnist_all.pkl", "bw") as fh:
    data = (train_imgs, 
            test_imgs, 
            train_labels,
            test_labels,
            #train_labels_one_hot,
            #test_labels_one_hot
            )
    pickle.dump(data, fh)
    
# Load images:
with open(data_path+"mnist_all.pkl", "br") as fh:
    data = pickle.load(fh)


train_imgs = data[0]
test_imgs = data[1]
train_labels = data[2]
test_labels = data[3]
train_labels_one_hot = data[4]
test_labels_one_hot = data[5]

train_imgs[2].shape










