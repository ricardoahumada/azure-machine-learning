# import libraries
import mlflow
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import pyspark.pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression


def main(args):
    # Inicializar Spark
    # spark = SparkSession.builder.appName("DiabetesTraining").getOrCreate()
    # spark = SparkSession.getActiveSession()
    spark = SparkSession.builder.getOrCreate()

    if spark is None:
        raise ValueError("No hay una sesión Spark activa disponible.")

    # read data
    df = get_data(args.input_data, spark)

    # split data
    X_train, X_test, y_train, y_test = split_data(df)

    # train model
    model = train_model(args.reg_rate, X_train, X_test, y_train, y_test)

    # evaluate model
    eval_model(model, X_test, y_test)

# function that reads the data
def get_data(path, spark):
    print("Reading data...")
    df = spark.read.csv(path, header=True, inferSchema=True)
    
    return df

# function that splits the data
def split_data(df):
    print("Splitting data...")
    # (trainingData, testData) = df.randomSplit([0.8, 0.2])

    # X_train, y_train = trainingData[['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness',
    # 'SerumInsulin','BMI','DiabetesPedigree','Age']].values, trainingData['Diabetic'].values
    
    # X_test, y_test = testData[['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness',
    # 'SerumInsulin','BMI','DiabetesPedigree','Age']].values, testData['Diabetic'].values
    X, y = df[['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness',
    'SerumInsulin','BMI','DiabetesPedigree','Age']].values, df['Diabetic'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

    return X_train, X_test, y_train, y_test

# function that trains the model
def train_model(reg_rate, X_train, X_test, y_train, y_test):
    mlflow.log_param("Regularization rate", reg_rate)
    print("Training model...")
    model = LogisticRegression(C=1/reg_rate, solver="liblinear").fit(X_train, y_train)

    return model

# function that evaluates the model
def eval_model(model, X_test, y_test):
    # calculate accuracy
    y_hat = model.predict(X_test)
    acc = np.average(y_hat == y_test)
    print('Accuracy:', acc)
    mlflow.log_metric("training_accuracy_score", acc)

    # calculate AUC
    y_scores = model.predict_proba(X_test)
    auc = roc_auc_score(y_test,y_scores[:,1])
    print('AUC: ' + str(auc))
    mlflow.log_metric("AUC", auc)


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--input_data", type=str, required=True, help="Path to the input dataset")
    parser.add_argument("--reg_rate", dest='reg_rate', type=float, default=0.01)

    # parse args
    args = parser.parse_args()

    # return args
    return args

# run script
if __name__ == "__main__":
    # add space in logs
    print("\n\n")
    print("*" * 60)

    # parse args
    args = parse_args()

    # run main function
    main(args)

    # add space in logs
    print("*" * 60)
    print("\n\n")
