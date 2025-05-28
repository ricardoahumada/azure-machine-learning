# Databricks notebook source
# MAGIC %md
# MAGIC # Ejecutar un script de entrenamiento en Spark (Databrocs)

# COMMAND ----------

from pyspark.sql import SparkSession

# Inicializar SparkSession
spark = SparkSession.builder.getOrCreate()

print("Clúster de Spark inicializado.", spark)

# COMMAND ----------

# import libraries
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import pyspark.pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

# COMMAND ----------

# function that reads the data
def get_data(path):
    print("Reading data...")
    df = spark.read.csv(path, header=True, inferSchema=True)
    
    return df

# function that splits the data
def split_data(df):
    print("Splitting data...")
    (trainingData, testData) = df.randomSplit([0.8, 0.2])

    return trainingData, testData

# function that trains the model
def train_model(train_data, test_data):
    print("Training model...")

    assembler = VectorAssembler(inputCols=['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness', 'SerumInsulin','BMI','DiabetesPedigree','Age'],outputCol="features")
    
    log_reg = LogisticRegression(featuresCol='features', labelCol='Diabetic')

    # Creating the pipeline
    pipeline = Pipeline(stages=[assembler, log_reg])

    fit_model = pipeline.fit(train_data)

    return fit_model

# function that evaluates the model
def eval_model(model, test_data):
    # Storing the results on test data
    results = model.transform(test_data)
    
    # Calling the evaluator
    res = BinaryClassificationEvaluator(rawPredictionCol='prediction',labelCol='Diabetic')

    # Accuracy, Precision, and Recall
    multi_evaluator = MulticlassClassificationEvaluator(labelCol="Diabetic", predictionCol="prediction")
    accuracy = multi_evaluator.evaluate(results, {multi_evaluator.metricName: "accuracy"})
    precision = multi_evaluator.evaluate(results, {multi_evaluator.metricName: "weightedPrecision"})
    recall = multi_evaluator.evaluate(results, {multi_evaluator.metricName: "weightedRecall"})
    
    print('accuracy:', accuracy)
    print('precision:', precision)
    print('recall:', recall)


    # Evaluating the AUC on results
    ROC_AUC = res.evaluate(results)
    print('ROC_AUC:', ROC_AUC)


# COMMAND ----------

spark = SparkSession.getActiveSession()

# read data
df = get_data('file:/Workspace/diabetes/diabetes.csv')

# COMMAND ----------

# split data
train_data, test_data = split_data(df)


# COMMAND ----------

# # train model
model = train_model(train_data, test_data)


# COMMAND ----------

# # evaluate model
eval_model(model, test_data)