

# Create a pipeline that standardizes the data then creates a model
from pandas import read_csv
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(url, names=names)
dataframe.shape
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
# create pipeline
estimators = []
# to convert different units to same units for comparision
estimators.append(('standardize', StandardScaler()))
# for dimensionality reduction, closely related to PCA
estimators.append(('lda', LinearDiscriminantAnalysis()))
model = Pipeline(estimators)
# evaluate pipeline


results = cross_val_score(model, X, Y, cv=10)
print(results)
print(results.mean())