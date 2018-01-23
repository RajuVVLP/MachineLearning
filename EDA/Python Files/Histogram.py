import pandas as pd
import os
from sklearn import preprocessing
import matplotlib.pyplot as plt
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

#returns current working directory
os.getcwd()
#changes working directory
os.chdir("D:/Data Science/Data/")
titanic_train = pd.read_csv("train.csv")

#EDA
titanic_train.shape
titanic_train.info()

titanic_train.loc[0:6,'Age']

titanic_train[['Age']].info() #Now you notice Age columns has MISSING data
age_imputer = preprocessing.Imputer()

#Fit the imputer on X.
age_imputer.fit(titanic_train[['Age']])
#.transform(): Impute all missing values in X.
titanic_train[['Age']] = age_imputer.transform(titanic_train[['Age']])
print(sorted(titanic_train.loc[0:20,'Age'], key=int))
plt.hist(titanic_train.loc[0:20,'Age'],5)
plt.show()