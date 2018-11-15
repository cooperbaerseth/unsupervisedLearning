import numpy as np
from sklearn.datasets import load_iris
from keras.datasets import mnist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

def get_var_explained(data, n_comp, rand_state):
    pca = PCA(n_components=n_comp, random_state=77).fit(data)
    print(pca.explained_variance_ratio_)

data = load_iris()

k_means = KMeans(n_clusters=3, random_state=77).fit(data.data)
plt.figure(); plt.title("K-Means Clustered: Raw Data")
plt.scatter(data.data[:,2], data.data[:,3], c=k_means.predict(data.data), cmap='plasma')

pca = PCA(n_components=2, random_state=77).fit_transform(data.data)
plt.figure(); plt.title("Projected Data")
plt.scatter(pca[:,0], pca[:,1])

k_means_pca = KMeans(n_clusters=3, random_state=77).fit(pca)
plt.figure(); plt.title("K-Means Clustered: Projected Data")
plt.scatter(pca[:,0], pca[:,1], c=k_means_pca.predict(pca), cmap='plasma')



data = load_iris()
(x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = mnist.load_data()
X_train_mnist = x_train_mnist.reshape((x_train_mnist.shape[0], pow(x_train_mnist.shape[1], 2)))

get_var_explained(X_train_mnist, 10, 77)








plt.pause(0.05)
raw_input('Press Enter to exit')