from sklearn import decomposition
import seaborn as sns
import pandas as pd

#Highly correlated columns (X1, X2)
df1 = pd.DataFrame({'x1':[10, 20, 30, 40],'x2':[15, 25, 35, 45], })
sns.jointplot('x1','x2',df1)
pca = decomposition.PCA(n_components=2)
pca.fit(df1)
pca.components_[0]
##understand how much variance captured by each principal component
print(pca.explained_variance_)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.cumsum())
df1_pca = pca.transform(df1)

#Not much correlated
df2 = pd.DataFrame({'x1':[10, 20, 30, 40],'x2':[100, 5000, 1, 5]})
sns.jointplot('x1','x2',df2)
pca = decomposition.PCA(n_components=1)
pca.fit(df2)
pca.components_[0]
#understand how much variance captured by each principal component
print(pca.explained_variance_)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.cumsum())
df2_pca = pca.transform(df2)
print(df2_pca)