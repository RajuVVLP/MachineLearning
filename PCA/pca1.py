import numpy as np
from sklearn import decomposition #PCA Package
import pandas as pd

#3 features with 5 records
df1= pd.DataFrame({
        'F1':[10,2,8,9,12],
        'F2':[20,5,17,20,22],
        'F3':[10,2,7,10,11]}) #F1, F2, F3... Are features
pca = decomposition.PCA(n_components=3) #n_components=3 means, transform the data to 3 dimensions.
#find eigen values and eigen vectors of covariance matrix of df
#.fit builds pca model for given fetures to prinicpal components
#Equation: 
#PC1 = F1*w11+F2*w12+F3*1w13.....
#PC2 = F1*w21+F2*w22+F3*w23.....
#PC3 = F1*w31+F2*w32+F3*w33.....
pca.fit(df1)
print(pca.components_)
#convert all the data points from standard basis to eigen vector basis
df1_pca = pca.transform(df1)

#variance of data along original axes
np.var(df1.F1) + np.var(df1.F2) + np.var(df1.F3)
#variance of data along principal component axes
#show eigen values of covariance matrix in decreasing order
np.sum(pca.explained_variance_)

#understand how much variance captured by each principal component
print(pca.explained_variance_)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.cumsum())

#show the principal components
#show eighen vectors of covariance matrix of df
pca.components_[0]
pca.components_[1]
pca.components_[2]

#specify number of required dimensions as n_components
pca = decomposition.PCA(n_components=2)
pca.fit(df1)
pca.explained_variance_
pca.components_[0]
pca.components_[1]
df1_pca = pca.transform(df1)
print(df1_pca)