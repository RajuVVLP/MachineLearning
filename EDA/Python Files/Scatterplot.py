
import os
import pandas as pd
import matplotlib.pyplot as plt


#returns current working directory
os.getcwd()
#changes working directory
os.chdir("D:/Data Science/Data")

titanic_train = pd.read_csv("train.csv")

#EDA
titanic_train.shape
titanic_train.info()

titanic_train1 = pd.get_dummies(titanic_train, columns=['Pclass', 'Sex', 'Embarked'])
titanic_train1.shape
titanic_train1.info()
titanic_train1.head(6)

print(titanic_train.loc[0:10,'PassengerId'])
print(titanic_train.loc[0:10,'SibSp'])

#No correlation
plt.scatter(titanic_train.loc[0:10,'PassengerId'], titanic_train.loc[0:10,'SibSp'])
plt.title('Scatter plot')
plt.xlabel('PassengerId')
plt.ylabel('SibSp')
plt.show()


#positive correlation
df = pd.DataFrame({'id':[1,2,3], 'fare':[13, 18, 25]})
plt.scatter(df['id'], df['fare'])
plt.title('Scatter plot')
plt.xlabel('id')
plt.ylabel('fare')
plt.show()

#inverse correlation
df = pd.DataFrame({'id':[1,2,3], 'fare':[13, 12, 11]})
plt.scatter(df['id'], df['fare'])
plt.title('Scatter plot')
plt.xlabel('id')
plt.ylabel('fare')
plt.show()



# =============================================================================
# import seaborn as sns
# os.chdir("D:\Shared\EDA\House")
# def viz_cont_cont(df, features, target):
#     for feature in features:
#         sns.jointplot(x = feature, y = target, data = df)
# 
# house_train = pd.read_csv("house_train.csv")
# target = 'SalePrice'
# features = ['BsmtUnfSF','1stFlrSF', '2ndFlrSF']
# viz_cont_cont(house_train, features, target)
# =============================================================================

