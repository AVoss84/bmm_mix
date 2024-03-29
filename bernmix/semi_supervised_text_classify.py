import os, pickle
import numpy as np
from numpy import log, sum, exp, prod
from numpy.random import beta, binomial, dirichlet, uniform, gamma, seed, multinomial, gumbel, rand
from imp import reload
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm.auto import tqdm
#import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from bernmix.utils import bmm_utils as bmm
from bernmix.utils import utils

reload(bmm)

# Example:
corpus = np.array([['I', 'like', 'cats'], ['I', 'like', 'dogs'], ['What', 'do', 'you'], ['I', 'like', 'cats']])
corpus
labels = np.array([2,6, 2, 2])

nb = utils.naiveBayes()
class_prior_prob, class_cond_prob = nb.fit(corpus, labels)
class_cond_prob
class_prior_prob
np.exp(nb.log_pd_theta)

#----------------------

corpus = np.array(['I like cats', 'I like dogs', 'What do you like?', 'This is fun!!!'])
y = np.array([2,6, 2, 2])

nb = utils.make_nb_feat().fit(corpus,y)

np.exp(nb.log_cond_distr_train.T)
nb.joint_abs_freq_train.T
nb.model.class_count_

for i in dir(nb.model):
    print(i)

#nb.model.get_params()

#------------------------
N = 10**5
K = 3           # number of mixture components
D = 50           # dimensions / number of features     

alphas = gamma(shape=5, size=K)               # shape parameter

print(sum(alphas))                              # equivalent sample size

p_true = dirichlet(alpha = alphas, size = 1)[0]
p_true
theta_true = beta(a = .7, b = .9, size = K*D).reshape(D,K)


# Generate data from mixture model:
#------------------------------------
# Draw from Bernoulli:
#probs = np.random.uniform(size=10000)
#rbern = (np.random.uniform(size=D) < mu_k[:,Z[2]]) * 1
X, Z = bmm.sample_bmm(N, p_true, theta_true)

latent_true = np.argmax(Z,1)          # true cluster assignements    

X.shape
Z.shape

# Set starting values for parameters:
#----------------------------------------
#seed(12)

#K = 10           # number of mixture components
D = X.shape[1]

alphas = gamma(shape=2, size=K)               # Dirichlet hyperparameters -> concentration param.
print(sum(alphas))

p_0 = dirichlet(alpha = alphas, size = 1)[0]
#p_0 = np.array([1/K]*K)  
theta_0 = beta(a = 1, b = 1, size = K*D).reshape(D,K)

#----------
# Run EM:    
#----------
logli, p_em, theta_em, latent_em = bmm.mixture_EM(X = X, p_0 = p_0, theta_0 = theta_0, n_iter = 500, stopcrit = 10**(-3))

#----------------
# Plot loglike.:
#----------------
burn_in = 5

plt.plot(logli[burn_in:], 'b--')
plt.title("Convergence check")
plt.xlabel('iterations')
plt.ylabel('loglikelihood')
plt.show()

# Compare with ground truth:
#---------------------------
print(p_em)
print(p_true)

theta_em
theta_true

confusion_matrix(latent_true, latent_em)


##################
# Gibbs sampler
##################
seed(12)

MC = 2000        # Monte Carlo runs
burn_in = 500    # discard those draws for burn-in

#K = 3
N, D = X.shape[0], X.shape[1]

p_draws = np.empty((MC,K))                                  # mixture weights draws
theta_draws = np.empty((MC,X.shape[1],K))                   # theta success rates 
latent_draws = np.empty((MC,N))                             # latent variable draws, Z

alphas = gamma(shape=2, size=K)               # shape parameters
p_0 = dirichlet(alpha = alphas, size = 1)[0]
#p_0 = np.array([1/K]*K)
theta_0 = beta(a = 1.3, b = 1.7, size = K*D).reshape(D,K)

p_draws[0,:], theta_draws[0,:,:] = p_0, theta_0 

gammas, deltas = gamma(shape=1.5, size=K), rand(K)     # uniform random draws   

#----------------------------
# Sample from full cond.:
#----------------------------
for i in tqdm(range(1,MC)):           

    latent_draws[i,:], p_draws[i,:], theta_draws[i,:,:] = bmm.gibbs_pass(p_draws[i-1,:], 
                                                      theta_draws[i-1,:,:], X, 
                                                      alphas = alphas, 
                                                      hyper_para = {'gammas': gammas, 'deltas': deltas})
print("Finished!")
#-----------------------------------------------------------------------------------------------------------

latent_draws.shape
p_draws.shape
theta_draws.shape

# Bayes estimates:
#---------------------
theta_bayes = np.mean(theta_draws[burn_in:,:, :],axis=0)
theta_bayes#.shape
theta_true

p_bayes = np.mean(p_draws[burn_in:,],axis=0)

print(p_bayes)
print(p_true)

latent_bayes = np.around(np.mean(latent_draws[burn_in:,:],axis=0))
latent_bayes.shape

# Compute performance metrics - 
# compare MAP estimates of cluster assignements with ground truth labels:
#-----------------------------------------------------------------------------
# Note: label switching issue when checking for simple accuracy!!

print(accuracy_score(latent_true, latent_bayes))    

confusion_matrix(latent_true, latent_bayes)


# Plot MCMC results:
#---------------------------------
plt.figure(figsize=(10, 20))
for j in range(p_draws.shape[1]):
    plt.subplot(5,2,j+1)
    plt.plot(p_draws[burn_in:, j]);
    plt.title('Trace for $p_{%d}$' % j)

plt.figure(figsize=(10, 20))
for j in range(theta_draws.shape[1]):
    plt.subplot(25,2,j+1)
    plt.plot(theta_draws[100:,j, 0]);
    plt.title('Trace for $theta_{%d}$' % j)


# Calculate ACF and PACF upto 50 lags
#acf_50 = acf(df.value, nlags=50)
#pacf_50 = pacf(df.value, nlags=50)

plt.figure(figsize=(10, 20))
for j in range(p_draws.shape[1]):
    fig, axes = plt.subplots(1,2,figsize=(16,3), dpi= 100)
    plot_acf(p_draws[burn_in:, j], lags=100, ax=axes[0])
    plt.xlabel("Lags"); 
    plt.title('ACF for $p_{%d}$' % j)
    plot_pacf(p_draws[burn_in:, j], lags=100, ax=axes[1])
    #plt.xlabel("Lags"); #plt.ylabel("PACF")
    plt.title('PACF for $p_{%d}$' % j)


#################### REAL DATA ############################
###########################################################

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale
from sklearn.pipeline import Pipeline


np.random.seed(42)

X_digits, y_digits = load_digits(return_X_y=True)
data = scale(X_digits)

X_digits.shape

n_samples, n_features = data.shape
n_digits = len(np.unique(y_digits))
labels = y_digits

sample_size = 300

print("n_digits: %d, \t n_samples %d, \t n_features %d"
      % (n_digits, n_samples, n_features))

pca = PCA(n_components=n_digits)
kmeans = KMeans(n_clusters=n_digits,n_init=1)
predictor = Pipeline([('pca', pca), ('kmeans', kmeans)])

predict = predictor.fit(data).predict(data)
predict

stats = bmm.clusters_stats(predict, labels)
purity = bmm.clusters_purity(stats)

print("Plotting an extract of the 10 clusters, overall purity: %f" % purity)

bmm.plot_clusters(predict, labels, stats, data)

############################################################
# MNIST
############################################################

image_size = 28                  # width and length
no_of_different_labels = 10 
image_pixels = image_size**2
image_pixels

# Read data in:
#os.getcwd()
data_path = "C:\\Users\\Alexander\\Documents\\Github\\bmm_mix\\bernmix\\data\\"

train_data = np.loadtxt(data_path + "mnist_train.csv", 
                        delimiter=",")

train_data.shape

test_data = np.loadtxt(data_path + "mnist_test.csv", 
                       delimiter=",") 

test_data.shape

test_data[:10]

test_data[test_data==255]
test_data.shape

#train_imgs = ((train_data[:, 1:]/255) > .5)*1.
train_imgs = np.asfarray(train_data[:, 1:])/255  # we avoid 0 values as inputs
test_imgs = np.asfarray(test_data[:, 1:])/255

X = train_imgs.copy()
X = test_imgs.copy()

fac = 0.99 / 255
train_imgs = np.asfarray(train_data[:, 1:]) * fac + 0.01  # we avoid 0 values as inputs which are capable of preventing weight updates
test_imgs = np.asfarray(test_data[:, 1:]) * fac + 0.01

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


# Log-sum trick:
x = np.arange(1,1000)

log(sum(exp(x)))

a = max(x) + log(sum(exp(x - max(x))));
a


################################################################



