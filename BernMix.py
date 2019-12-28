
import os, pickle, copy
import numpy as np
from numpy import log, sum, exp, prod
from numpy.random import beta, binomial, dirichlet, uniform, gamma, seed, multinomial
#from scipy.stats import multinomial
from imp import reload
import matplotlib.pyplot as plt

os.chdir("C:\\Users\\Alexander\\Documents\\\Github\\bmm_mix")

from bernmix.utils import bmm_utils as bmm
#os.getcwd()
reload(bmm)

#seed(12)

N = 1000
K = 3           # number of mixture components
D = 10           # dimensions / number of features     

alphas = gamma(shape=1, size=K)               # shape parameter
p_true = dirichlet(alpha = alphas, size = 1)[0]
p_true

theta_true = beta(a = .7, b = .9, size = K*D).reshape(D,K)


# Generate data from mixture model:
#------------------------------------
# Draw from Bernoulli:
#probs = np.random.uniform(size=10000)
#rbern = (np.random.uniform(size=D) < mu_k[:,Z[2]]) * 1
X, Z = bmm.sample_bmm(N, p_true, theta_true)

X.shape
Z.shape

# Set starting values for parameters:
#----------------------------------------
#seed(12)

#K = 10           # number of mixture components
D = X.shape[1]

#alphas = gamma(shape=1, size=K)               # shape parameters
#p_0 = dirichlet(alpha = alphas, size = 1)[0]
p_0 = np.array([1/K]*K)  # K>2
theta_0 = beta(a = 1, b = 1, size = K*D).reshape(D,K)


#----------
# Run EM:    
#----------
logli, p_em, theta_em = bmm.mixture_EM(X = X, p_0 = p_0, theta_0 = theta_0, n_iter = 200, stopcrit = 10**(-4))


#----------------
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



