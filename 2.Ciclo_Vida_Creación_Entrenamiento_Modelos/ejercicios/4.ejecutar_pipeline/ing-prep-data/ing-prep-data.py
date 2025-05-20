
# import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import argparse
from pathlib import Path

import mlflow


def main(args):
    df = read_data(args.input_data)
    df2 = preprocess_data(df)
    
    output_df = df2.to_csv((Path(args.output_data) / "housing_prep.csv"), index = False)



def read_data(uri):
    # 1. Cargar los datos
    data = pd.read_csv(uri)
    # 2. Inspeccionar las columnas
    print(data.columns)

    return data

def preprocess_data(data):
    # 3. Descartar columnas no significativas
    # Las columnas 'date', 'street', 'city', 'statezip', 'country' no son relevantes para la predicción
    columns_to_drop = ['date', 'street', 'city', 'statezip', 'country']
    data = data.drop(columns=columns_to_drop)

    return data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", dest='input_data', type=str)
    args = parser.parse_args()

    return args


# run script
if __name__ == "__main__":
    args = parse_args()
    main(args)
