
import os
import pandas as pd
import seaborn as sns
import numpy as np



#to uncover the mismatch of levels between train and test data
def merge(df1, df2):
    return pd.concat([df1, df2])

        
     #Visualization of continous columns
def viz_cont(df, features):
    for feature in features:
        sns.distplot(df[feature],kde=False)
        
        
os.chdir("D:/Data Science/House")

house_train = pd.read_csv("house_train.csv")
house_test = pd.read_csv("house_test.csv")

house_test['SalePrice'] = 0

house_data = merge(house_train, house_test)



#plot my sale price
x = house_train['SalePrice']
sns.distplot(x, kde=False)

#smoothing the values as the sale price is big  
house_train['log_sale_price'] = np.log(house_train['SalePrice'])
#see how the sale price looks with log
x = house_train['log_sale_price']
sns.distplot(x, kde=False)