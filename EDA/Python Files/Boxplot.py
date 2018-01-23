import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

#returns current working directory
os.getcwd()
#changes working directory
os.chdir("D:/Data Science/House/")
titanic_train = pd.read_csv("house_train.csv")
print(titanic_train.info())

def viz_cat_cont_box(df, features, target):
    for feature in features:
        sns.boxplot(x = feature, y = target,  data = df)
        plt.xticks(rotation=45)

target = 'SalePrice'
features = ['Neighborhood']
viz_cat_cont_box(titanic_train, features, target)



