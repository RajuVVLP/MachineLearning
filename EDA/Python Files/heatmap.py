import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def get_target_corr(corr, target):
    return corr[target].sort_values(axis=0,ascending=False)

def get_heat_map_corr(df):
    corr = df.select_dtypes(include = ['number']).corr()
    sns.heatmap(corr, square=True)
    plt.xticks(rotation=70)
    plt.yticks(rotation=70)
    return corr

os.chdir("D:/Data Science/House")

house_train = pd.read_csv("house_train.csv")
house_train.shape
house_train.info()

house_train['log_sale_price'] = np.log(house_train['SalePrice'])
house_train = house_train.loc[0:100,['MSSubClass','LotFrontage','LotArea','SalePrice']] 

#explore relation among all continuous features vs saleprice 

corr = get_heat_map_corr(house_train)

#get_target_corr(corr, 'SalePrice')
get_target_corr(corr, 'log_sale_price')