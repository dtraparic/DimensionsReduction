
import pickle # sauvegarder le dataset

from sklearn.datasets import fetch_openml
# from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
import matplotlib.pyplot as plot
# from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

np.set_printoptions(threshold=np.inf) # option pour les print de valeurs

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


# Maintenant, mnist = tout le dataset
# mnist.data.shape pour afficher (trainingset_length, testingset_length)
# mnist.target.shape pour afficher le nombre de labels

# On normalise les données d'entrainement entre 0 et 1

normaliseur = MinMaxScaler()
mnist.data = normaliseur.fit_transform(mnist.data)

# Nb images = 70000
# image = 28x28 = 784 pixels
print('dimensions des données : ', mnist.data.shape) # 70000 lignes et 784 colonnes (ou l'inverse)
print('nombre de pixels :', mnist.data[1].shape)
print('nombre de labels : ', mnist.target.shape)
# print(max(mnist.data[1]))
# print(mnist.data)

train = mnist.data[1:-100]
test = mnist.data[100:]
train_label = mnist.target[1:-100]
test_label = mnist.data[100:]

print(mnist.target)

nb_class = 10;
all_means = []
Sw = [] # covariance intra classe
Si = [] # covariance inter classe

for i in range(0, 9):

    mask_vector_i = (train_label == str(i)) # = [0 1 0 0 1...], une matrice d'indices pour selectionner toutes les données d'un label particulier
    # print(mask_vector_i[1:10])
    print(train.shape)
    print(train[mask_vector_i, :].shape)

    train_i = train[mask_vector_i, :] # data d'une classe
    # print(train_i[1:10])
    print(train_i.shape)
    n_i = train_i.shape[1]
    part_i = n_i / nb_class
    mean_i = np.mean(train_i)

    Si_i = np.cov(train_i, train_i); # Autocovariance de chaque classe

    Sw += Si_i * part_i

    print(Sw.shape)
    print(Si_i.shape)

    Sb += part_i * (mean_i - np.mean(train)) * np.transpose(mean_i - np.mean(train)) # ici on pourrait passer par la cov si on a un tableau des moyennes de classes

    print(n_i, mean_i)

    all_means.append(mean_i)
    all_Sw.append(Sw_i)
    all_Si.append(Si_i)

cov = np.cov(mnist.data[1])




plot.figure(figsize=(8,4));

# Original Image
plot.subplot(1, 2, 1);
plot.imshow(mnist.data[1].reshape(28,28),
              cmap = plot.cm.gray, interpolation='nearest',
              clim=(0, 1));
plot.xlabel('784 components', fontsize = 14)
plot.title('Original Image', fontsize = 20);

# 154 principal components
# plot.subplot(1, 2, 2);
# plot.imshow(x_lda[1].reshape(28, 28),
#               cmap = plot.cm.gray, interpolation='nearest',
#               clim=(0, 1));
# plot.xlabel('154 components', fontsize = 14)
# plot.title('95% of Explained Variance', fontsize = 20);

plot.show();
