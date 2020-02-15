
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plot
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

np.set_printoptions(threshold=np.inf)


mnist = fetch_openml('mnist_784', version=1, cache=True) # Choppe tout chez un serveur

# Matrice de covariance
matrice = np.array([[0, 2], [1, 1], [2, 0]])
MCov = np.cov(matrice)

# Diagonalisation
eigenvalues, eigenvectors = np.linalg.eig(MCov)
#eigenvectors.shape, avec le premier vecteur eigenvectors[:,0]



# Maintenant, mnist = tout le dataset
# mnist.data.shape pour afficher (trainingset_length, testingset_length)
# mnist.target.shape pour afficher le nombre de labels

# On normalise les données d'entrainement entre 0 et 1
normaliseur = MinMaxScaler()
mnist.data = normaliseur.fit_transform(mnist.data)

# print(mnist.data[1])
# print(max(mnist.data[1]))
# print(mnist.data)

# On crée un objet PCA, avec 95% de la variance assurée
pca_95 = PCA(.95)

# Supressions des dimensions les moins importantes
lower_dimensional_data = pca_95.fit_transform(mnist.data)
# Ici on remet les dimensions supprimées mais à 0 pour être au même nombre de dimensions
approximation = pca_95.inverse_transform(lower_dimensional_data)

# pca_95.n_components_ = le choix qu'a fait pca_95 pour arriver à 95 % de variance
#plot.figure(figsize=(8,4));

# Original Image
#plot.subplot(1, 3, 1);
#plot.imshow(mnist.data[1].reshape(28,28),
#              cmap = plot.cm.gray, interpolation='nearest',
#              clim=(0, 1));
#plot.xlabel('784 components', fontsize = 14)
#plot.title('Original Image', fontsize = 20);
#
## 154 principal components
#plot.subplot(1, 2, 2);
#plot.imshow(approximation[1].reshape(28, 28),
#              cmap = plot.cm.gray, interpolation='nearest',
#              clim=(0, 1));
#plot.xlabel('154 components', fontsize = 14)
#plot.title('95% of Explained Variance', fontsize = 20);
#
#plot.show();
