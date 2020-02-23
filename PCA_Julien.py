# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 17:50:27 2020

@author: julien
"""

import pandas as pd


from sklearn.preprocessing import StandardScaler


import numpy as np


df = pd.read_csv(
    filepath_or_buffer='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', 
    header=None, 
    sep=',')

df.columns=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
df.dropna(how="all", inplace=True) # drops the empty line at file-end

df.tail()

# split data table into data X and class labels y

X = df.iloc[:,0:4].values
y = df.iloc[:,4].values
           

import plotly as py

py.tools.set_credentials_file(username='jcc1234567890',api_key='FwyzbbivTXOGvMbVypkU')

# plotting histograms
data = []

legend = {0:False, 1:False, 2:False, 3:True}

colors = {'Iris-setosa': '#0D76BF', 
          'Iris-versicolor': '#00cc96', 
          'Iris-virginica': '#EF553B'}

for col in range(4):
    for key in colors:
        trace = dict(
            type='histogram',
            x=list(X[y==key, col]),
            opacity=0.75,
            xaxis='x%s' %(col+1),
            marker=dict(color=colors[key]),
            name=key,
            showlegend=legend[col]
        )
        data.append(trace)

layout = dict(
    barmode='overlay',
    xaxis=dict(domain=[0, 0.25], title='sepal length (cm)'),
    xaxis2=dict(domain=[0.3, 0.5], title='sepal width (cm)'),
    xaxis3=dict(domain=[0.55, 0.75], title='petal length (cm)'),
    xaxis4=dict(domain=[0.8, 1], title='petal width (cm)'),
    yaxis=dict(title='count'),
    title='Distribution of the different Iris flower features'
)

fig = dict(data=data, layout=layout)
py.plotly.iplot(fig, filename='exploratory-vis-histogram')





X_std = StandardScaler().fit_transform(X)

mean_vec = np.mean(X_std, axis=0)
cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)

print('Covariance matrix \n%s' %cov_mat)
print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)

for ev in eig_vecs:
    np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
print('Everything ok!')

# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort()
eig_pairs.reverse()

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('Eigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])
    
    

#
tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)




matrix_w = np.hstack((eig_pairs[0][1].reshape(4,1), 
                      eig_pairs[1][1].reshape(4,1)))

print('Matrix W:\n', matrix_w)

Y = X_std.dot(matrix_w)

