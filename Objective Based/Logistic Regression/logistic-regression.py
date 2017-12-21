#Objective based - Logistic Regression model(LR Model). It's for calssification data/problem
import os
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import linear_model 
from sklearn import model_selection

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

X_train = titanic_train1.drop(['PassengerId','Age','Cabin','Ticket', 'Name','Survived'], 1)
X_train.info()
y_train = titanic_train['Survived']

#Logistic Regression is available in sklearn-->liner_model
#random_state tells the algorithm to use same data sample. This avoids confusion when QA/UAT is performing the validation.
lr_estimator = linear_model.LogisticRegression(random_state=2017)
#Parameters: C - Control of overfit parameter/Regulizer(regularization strength), How much of L1/L2
#Penalty is nothing but L1 or L2
#max_iter: How many iteration to try... Being it uses Guess & Refine approach internally
#list(range(100,1000,200)-->Means starts with 100 till 1000 increment by 200 iterations.
lr_grid = {'C':list(np.arange(0.1,1.0,0.1)), 'penalty':['l1','l2'], 'max_iter':list(range(100,1000,200))}
#Remeber that we don't know which paramaeters for Controlling overfitting is good. We have keep try with different values,
#and check the diff between train score and CV score. Once they are close, then we can say that that's the best parameter.
#That means, you have to always compare train score and CV Score to come to optimal fitting
lr_grid_estimator = model_selection.GridSearchCV(lr_estimator, lr_grid, cv=10, n_jobs=1)
lr_grid_estimator.fit(X_train, y_train)
lr_grid_estimator.grid_scores_
final_model = lr_grid_estimator.best_estimator_
#To find out the best score
lr_grid_estimator.best_score_
#We get coefficients/un-knowns for each variable
#Finding oit coef_(s)/Features and intercept_/miniml value are only for our understanding. Actullay they will be used internally
final_model.coef_ #In this case we get 17 coefficients for 17 vairables
#intercept is nothing but minimal value. Bias in N/Nworks
final_model.intercept_

##Final Prections Preparation
titanic_test = pd.read_csv("test.csv")

#Note that you have to do the same work on test as well
#EDA
titanic_test.shape
titanic_test.info()

titanic_test1 = pd.get_dummies(titanic_test, columns=['Pclass', 'Sex', 'Embarked'])
titanic_test1.shape
titanic_test1.info()
titanic_test1.head(6)
titanic_test1.describe()

X_test = titanic_test1.drop(['PassengerId', 'Age','Cabin','Ticket', 'Name'], 1)
X_test.info()

#create an instance of Imputer class with required arguments
mean_imputer = preprocessing.Imputer()  #Default value for Imputer is mean
#compute mean of age and fare respectively
mean_imputer.fit(X_test[['Fare']])
#fill up the missing data with the computed means 
X_test[['Fare']] = mean_imputer.transform(X_test[['Fare']])

titanic_test['Survived'] = lr_grid_estimator.predict(X_test)

titanic_test.to_csv('submission.csv', columns=['PassengerId','Survived'],index=False)
