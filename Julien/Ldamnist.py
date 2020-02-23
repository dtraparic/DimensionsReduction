# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 10:46:02 2020

@author: julien
"""


from sklearn.datasets import fetch_openml
# from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
import matplotlib.pyplot as plot
# from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

np.set_printoptions(threshold=np.inf) # option pour les print de valeurs

mnist = fetch_openml('mnist_784', version=1, cache=True) # Choppe tout chez un serveur
# Maintenant, mnist = tout le dataset
# mnist.data.shape pour afficher (trainingset_length, testingset_length)
# mnist.target.shape pour afficher le nombre de labels

# On normalise les données d'entrainement entre 0 et 1
normaliseur = MinMaxScaler()
mnist.data = normaliseur.fit_transform(mnist.data)

print(mnist.data.shape);
# print(mnist.data[1])
# print(max(mnist.data[1]))
# print(mnist.data)

# On crée un objet PCA, avec 95% de la variance assurée
lda = LinearDiscriminantAnalysis(n_components=2)

# Supressions des dimensions les moins importantes
x_lda = lda.fit_transform(mnist.data)
lda.explained_variance_ratio_

plot.figure(figsize=(8,4));

# Original Image
plot.subplot(1, 2, 1);
plot.imshow(mnist.data[1].reshape(28,28),
              cmap = plot.cm.gray, interpolation='nearest',
              clim=(0, 1));
plot.xlabel('784 components', fontsize = 14)
plot.title('Original Image', fontsize = 20);

# 154 principal components
plot.subplot(1, 2, 2);
plot.imshow(x_lda[1].reshape(28, 28),
              cmap = plot.cm.gray, interpolation='nearest',
              clim=(0, 1));
plot.xlabel('154 components', fontsize = 14)
plot.title('95% of Explained Variance', fontsize = 20);

plot.show();