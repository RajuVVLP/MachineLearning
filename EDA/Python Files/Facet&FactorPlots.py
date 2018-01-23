import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt

#changes working directory
os.chdir("D:/Data Science/Data")
titanic_train = pd.read_csv("train.csv")

#FactorPlot
#create title column from name
def extract_title(name):
     return name.split(',')[1].split('.')[0].strip()
titanic_train['Title'] = titanic_train['Name'].map(extract_title)
titanic_train['Title']

sns.factorplot(x="Title", hue="Survived", data=titanic_train, kind="count", size=6)

titanic_train1 = titanic_train.loc[titanic_train['SibSp'] == 2]
#titanic_train1 = titanic_train.loc[titanic_train['SibSp'].isin(['0','2'])]
titanic_train1 = titanic_train.loc[titanic_train['SibSp'].isin(['0','2','5'])]

#FacetGrid
g = sns.FacetGrid(titanic_train1, col="SibSp",  row="Survived")
g = g.map(plt.hist, "SibSp")