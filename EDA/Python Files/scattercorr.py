import os
import pandas as pd
import matplotlib.pyplot as plt

def viz_cont_cont(df, features, target):
    for feature in features:
        plt.scatter(x = feature, y = target, data = df)
        

os.chdir("D:/Data Science/House")

house_train = pd.read_csv("house_train.csv")
house_train.shape
house_train.info()
        
target =  house_train['SalePrice']
#explore relationship of few columns against saleprice
features = ['BsmtUnfSF']
viz_cont_cont(house_train, features, target)

features = ['1stFlrSF']
viz_cont_cont(house_train, features, target)

features = ['2ndFlrSF']
viz_cont_cont(house_train, features, target)


