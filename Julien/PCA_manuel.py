#!/usr/bin/env python
# coding: utf-8

# # Diminution de dimensions par LDA puis classification 
# ## Dataset MNIST

# In[1]:


#Sur le code PCA il faut reduire toutes les données

import pickle # sauvegarder le dataset

from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib.pyplot as plot
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


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

mean_vec = np.mean(mnist.data, axis=0)
cov_mat = (mnist.data - mean_vec).T.dot((mnist.data - mean_vec)) / (mnist.data.shape[0]-1)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)

#print('Covariance matrix \n%s' %cov_mat)
#print('Eigenvectors \n%s' %eig_vecs)
#print('\nEigenvalues \n%s' %eig_vals)
#
#for ev in eig_vecs:
#   np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
#print('Everything ok!')

eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

#print(eig_pairs)
# Sort the (eigenvalue, eigenvector) tuples from high to low
print("aazdqzqsddqsds")
#print(eig_pairs[1])

sorted(eig_pairs, key=lambda x:x[0])

#print('Eigenvalues in descending order:')
#for i in eig_pairs:
#    print(i[0])
    
# il faut ensuite choisir une variance et recuperer dans la liste l'endroit ou il y a 
#plus d'un certain nombre de composante qui correspond a une certaine variance

tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

conf= 95
compteur=0
for i in cum_var_exp:
    compteur += 1
    if i>conf:
        break
    
print(compteur)
eig_pairs =eig_pairs[0:compteur]
#il faut choisir les x premier couple de valuer 
#matrix_w = np.hstack(eig_pairs[:][1].reshape(784,1))

l=[]
for i in eig_pairs:
    
    l.append(i[1])

l1 = np.reshape(l, (compteur, 784)).T

y = mnist.data.dot(l1)


#
## On print pour bien tout visualiser, et on s'occupe aussi de séparer les données d'entrainement des données de test, puis on fait la même chose en fonction des libellés.
#
## In[3]:
#
#
## Nb images = 70000
## image = 28x28 = 784 pixels
#
#print('dimensions des données : ', mnist.data.shape) # 70000 lignes et 784 colonnes (ou l'inverse)
#print('nombre de pixels :', mnist.data[1].shape)
#print('nombre de labels : ', mnist.target.shape)
## print(max(mnist.data[1]))
## print(mnist.data)
#
#DEBUT_TESTS = 60001;
#nb_train_voulu = 10000 #inférieur ou égal à 60k
#nb_test_voulu = 100 #inférieur ou égal à 10k 
#
#nb_test_ech = 10000;
#train = np.transpose(mnist.data[:nb_train_voulu])
#train_label = np.transpose(mnist.target[:nb_train_voulu])
#
#test = np.transpose(mnist.data[DEBUT_TESTS:DEBUT_TESTS+nb_test_voulu])
#test_label = np.transpose(mnist.target[DEBUT_TESTS:DEBUT_TESTS+nb_test_voulu])
#
#
#
#print("Test labels : ", test_label.min(),"->", test_label.max())
#print("Train labels : ", train_label.min(),"->", train_label.max())
#
#print(train.shape)
#print(test.shape)
#
#
## Les étapes principales de cet algorithme vont commencer ici. Cette boucle for va établir les covariances interclasses et intraclasses, pour chacune d'entre-elles, après avoir rassemblé qualitativement les échantillons.
## part_i correspond à un pourcentage de classe qui va jouer le rôle d'un coefficient d'importance.
## 
## On essaie différentes manières de calculer l'autocovariance : la méthode Si_i2 et Si_i3 fournissent les mêmes résultats et sont justes.
#
## In[4]:
#
#
#N = train.shape[1];
#dimension = train.shape[0]
#
#all_means = np.mean(train, axis=1)
#Sw = np.zeros((dimension, dimension))
#Sb = np.zeros((dimension, dimension))
#Si = [] # covariance inter classe
