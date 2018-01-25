# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 15:10:56 2018

@author: kavya.chitta
"""

import os
import inspect
import sys
import pyspark


# Initiate Spark context.
from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.sql import SQLContext
from pyspark.sql.types import * 
import pyspark.sql.functions as fn 
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorIndexer, VectorAssembler, IndexToString
from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml import Pipeline
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.functions import lit


SPARK_HOME=os.path.dirname(inspect.getfile(SparkContext))

sys.path.insert(0,os.path.join(SPARK_HOME,"python"))
sys.path.insert(0,os.path.join(SPARK_HOME,"python","lib"))
sys.path.insert(0,os.path.join(SPARK_HOME,"python","lib","pyspark.zip"))
sys.path.insert(0,os.path.join(SPARK_HOME,"python","lib","py4j-0.9-src.zip"))

# Configure Spark Settings
conf=SparkConf()
conf.set("spark.executor.memory", "1g")
conf.set("spark.cores.max", "2")
conf.setAppName("spark")

# Initialize SparkContext. 
sc = SparkContext('local', conf=conf)

#Define variables
csvFormat = "com.databricks.spark.csv"
PATH = 'D:/Data-Science/Titanic'

#Initiate SQLContext and read the train and test files
sqlContext = SQLContext(sc)
train = (sqlContext.read.format(csvFormat).load(PATH+'/train.csv',header=True,inferSchema=True))
test = (sqlContext.read.format(csvFormat).load(PATH+'/test.csv',header=True,inferSchema=True))

test.printSchema()
test.count()

#We dont find the Survived column in Test data and so in order to merge both, Survived column is added.
test = test.withColumn('Survived', lit(-1))

#the column order should match when merging and hence select is used with Test Data and merged with Train
result = train.union(test.select("PassengerId","Survived","Pclass","Name","Sex","Age","SibSp","Parch","Ticket","Fare","Cabin","Embarked"))
result.count()

#register with temp table and use SQL query to select records
result.registerTempTable('train_tmp')
sqlContext.sql('select Survived from train_tmp where Survived == -1').count()

test.count()

#Adding FamilySize as a new column
result = result.withColumn('FamilySize', train.SibSp+train.Parch+1)
result.printSchema()

#Fill nulls in Age and Fare columns with the mean of the column
result.registerTempTable('train_tmp')
sqlContext.sql('select Age from train_tmp where Age is null').show()

#A user defined function to fill the nulls with mean
def fill_with_mean(df, include=set()): 
    stats = df.agg(*(
        fn.avg(c).alias(c) for c in df.columns if c in include
    ))
    return df.fillna(stats.first().asDict())

result = fill_with_mean(result, ["Age","Fare"])

result.registerTempTable('train_tmp')
sqlContext.sql('select Age from train_tmp where Age is null').show()

result.registerTempTable('train_tmp')
sqlContext.sql('select Embarked from train_tmp where Embarked is null').show()

#Define a function mode in functions which are imported from pyspark.sql package
@fn.udf
def mode(x):
    from collections import Counter
    return Counter(x).most_common(1)[0][0]

#A user defined function to fill the nulls with mode
def fill_with_mode(df, include=set()): 
    stats = df.agg(*(
        mode(fn.collect_list(c)).alias(c) for c in df.columns if c in include
    ))
    return df.fillna(stats.first().asDict())

result.printSchema()
result = fill_with_mode(result, ["Embarked"])

result.registerTempTable('train_tmp')
sqlContext.sql('select Embarked from train_tmp where Embarked is null').show()

#Divide the Train and Test for model tuning and prediction
train_new = sc.parallelize(result.head(train.count())).toDF()
test_new = sqlContext.sql('select * from train_tmp where Survived == -1')
test_new = test_new.drop('Survived')

train_new.count()
test_new.count()
train_new.printSchema()
test_new.printSchema()

#Defining the string indexers for indexing categorical features -- PClass,Sex and Embarked
PClassIndexer = StringIndexer(inputCol="Pclass", outputCol="Pclass_Indexed").fit(train_new)
SexIndexer = StringIndexer(inputCol="Sex", outputCol="Sex_Indexed").fit(train_new)
EmbarkedIndexer = StringIndexer(inputCol="Embarked", outputCol="Embarked_Indexed").fit(train_new)

#We also index our label which corresponds to the Survived column
labelIndexer = StringIndexer(inputCol="Survived", outputCol="SurvivedIndexed").fit(train_new)
type(labelIndexer)

#All feature columns are to be assembled into one single column containing a vector regrouping all features.
numericFeatColNames = ["Age", "Fare", "FamilySize"]
idxdCategoricalFeatColName = ["Pclass_Indexed","Sex_Indexed","Embarked_Indexed"]
allIdxdFeatColNames = numericFeatColNames + idxdCategoricalFeatColName

#Using VectorAssembler to group all features into a single vector
assembler = VectorAssembler().setInputCols(allIdxdFeatColNames).setOutputCol("Features")

#Definind a DecisionTree Classifier with the column to be predicted-SurvivedIndexed and features-Features as inputs.
dt = DecisionTreeClassifier(labelCol="SurvivedIndexed", featuresCol="Features")
 
#The prediction is in indexed form and hence it is reversed using IndexToString
labelConverter = IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)

#Bring everything together using a pipeline. Every operation is executed in the sequence mentioned.
pipeline = Pipeline(stages=[PClassIndexer,SexIndexer,EmbarkedIndexer, labelIndexer,assembler, dt,labelConverter])

#Best model is selected from the parameter grid mentioned based on the average metrics calculated from the given no of folds.
paramGrid = ParamGridBuilder().addGrid(dt.maxBins, [10]).addGrid(dt.maxDepth, [4]).addGrid(dt.impurity, ["entropy", "gini"]).build()

#This is used to evaluate the model based on metrics. There are 3 built-in evaluators - For Regression, For Binary Classification and the other for Multiclass Classification.
#The ROC Curve is the default metric for the Binary Classification Evaluator.
evaluator = BinaryClassificationEvaluator().setLabelCol("SurvivedIndexed")

#An estimator is to be trained so that it could be used further for prediction.
#The estimator uses pipeline for training the model.
cv = CrossValidator().setEstimator(pipeline).setEvaluator(evaluator).setEstimatorParamMaps(paramGrid).setNumFolds(10)

#CrossValidator is an Estimator, we can obtain the best model for our data by calling the fit method on it.
crossValidatorModel = cv.fit(train_new)

crossValidatorModel.avgMetrics

#The prediction is made on test data using the estimator.
predictions = crossValidatorModel.transform(test_new)

type(predictions)
predictions.printSchema()

#PassengerId and Survived columns are selected out of the prediction and saved into a new Dataframe.
prediction_new = predictions.withColumn("Survived", predictions.predictedLabel).select("PassengerId", "Survived")

prediction_new.printSchema()

#The new dataframe is written to a csv file in Kaggle Submission format. Along with the csv file, 'crc' files are also created which are the metadata.
prediction_new.repartition(1).write.format('com.databricks.spark.csv').option("header", "true").save(PATH+'/Submission')

 #Stop the Spark context which is started.
sc.stop()