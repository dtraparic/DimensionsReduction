# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 11:47:49 2020

@author: julien
"""
#
#import numpy as np
#import matplotlib.pyplot as plt
#from matplotlib import offsetbox
#from sklearn import manifold, datasets, decomposition, discriminant_analysis
#from mpl_toolkits.mplot3d import Axes3D
# 
#digits = datasets.load_digits()
#X = digits.data
#y = digits.target
#n_samples, n_features = X.shape
# 
import pickle # sauvegarder le dataset

from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import offsetbox
from sklearn import manifold, datasets, decomposition, discriminant_analysis
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import MinMaxScaler




np.set_printoptions(threshold=np.inf) # option pour les print de valeurs


# On gère un système de cache grâce à pickle pour ne pas refetch les données à chaque run du programme, puis on normalise

# In[2]:


try:
    print("Searching for cached dataset...")
    mnist = pickle.load(open("mnist.pickle", "rb"))
    print("Cache found.")
except (OSError, IOError) as e: # ("ask for forgiveness" principle)
    print("No cache found. Downloading dataset...")
    mnist = fetch_openml('mnist_784', version=1, cache=True) # Choppe tout chez un serveur
    print("Saving dataset...")
    pickle.dump(mnist, open("mnist.pickle", "wb"))
    print("Saved.")
    
# On normalise les données d'entrainement entre 0 et 1

normaliseur = MinMaxScaler()
mnist.data = normaliseur.fit_transform(mnist.data)


# On print pour bien tout visualiser, et on s'occupe aussi de séparer les données d'entrainement des données de test, puis on fait la même chose en fonction des libellés.

# In[3]:


# Nb images = 70000
# image = 28x28 = 784 pixels

print('dimensions des données : ', mnist.data.shape) # 70000 lignes et 784 colonnes (ou l'inverse)
print('nombre de pixels :', mnist.data[1].shape)
print('nombre de labels : ', mnist.target.shape)
# print(max(mnist.data[1]))
# print(mnist.data)

DEBUT_TESTS = 60001;
nb_train_voulu = 1797 #inférieur ou égal à 60k
nb_test_voulu = 100 #inférieur ou égal à 10k 

nb_test_ech = 10000;

train = mnist.data[:nb_train_voulu]
train_label = mnist.target[:nb_train_voulu]
train_label = train_label.astype(np.int)
n_samples, n_features = train.shape

def embedding_plot(X, title):
    x_min, x_max = np.min(X, axis=0), np.max(X, axis=0)
    X = (X - x_min) / (x_max - x_min)
 
    plt.figure()
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(X[:,0], X[:,1], lw=0, s=40, c=color)
    shown_images = np.array([[1., 1.]])

#    for i in range(X.shape[0]):
#        if np.min(np.sum((X[i] - shown_images) ** 2, axis=1)) < 1e-2: continue
#        shown_images = np.r_[shown_images, [X[i]]]
#        ax.add_artist(offsetbox.AnnotationBbox(offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r), X[i]))
#
#    plt.xticks([]), plt.yticks([])
    plt.title(title)
 
def embedding_plot_3d(X, title):
    x_min, x_max = np.min(X, axis=0), np.max(X, axis=0)
    X = (X - x_min) / (x_max - x_min)
    
    plt.figure()
    ax = plt.subplot(111, projection='3d')
    sc = ax.scatter(X[:,0], X[:,1], X[:,2], lw=0, s=10, c=color)
    
    plt.title(title)

colors = ['mediumvioletred','indigo','blueviolet','navy','seagreen','darkgreen','yellowgreen','greenyellow','yellow','orangered']
color = [colors[i] for i in train_label]

	
X_pca = decomposition.PCA(n_components=2).fit_transform(train)
#embedding_plot_3d(X_pca, "PCA dans un espace réduit à trois dimensions")
embedding_plot(X_pca, "PCA dans un espace réduit à deux dimensions")
plt.show()


	
X_lda = discriminant_analysis.LinearDiscriminantAnalysis(n_components=2).fit_transform(train, train_label)
#embedding_plot_3d(X_lda, "LDA dans un espace réduit à trois dimensions")
embedding_plot(X_lda, "LDA dans un espace réduit à deux dimensions")
 
plt.show()