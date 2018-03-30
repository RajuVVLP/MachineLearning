
import os
import inspect
import sys


# Initiate Spark context.
from pyspark import SparkContext
from pyspark import SparkConf

#
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

lines=sc.textFile("D:/Samples/Python/DataScience/House/house_test.csv")

print (lines.count())


sc.stop()
