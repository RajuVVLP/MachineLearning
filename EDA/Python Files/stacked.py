
#example 1
import pandas as pd
import os
os.chdir("D:/Data Science/House")

train = pd.read_csv("house_train.csv")

train.shape

a = train.loc[0:5,'MSSubClass']
b = train.loc[0:5,'LotFrontage']
c = train.loc[0:5,'SalePrice']

df = pd.DataFrame({"SubClass":a,
              "Frontage":list(b), "Price":c})

df.pivot("SubClass","Frontage").plot(kind="bar", stacked=True)

