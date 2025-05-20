
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
    model, X_test, y_test = train_model('price', df2, 0.3)
    evaluate_model(model, X_test, y_test)


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


def train_model(target, data, test_size=0.2):
    # 4. Separar características (X) y objetivo (y)
    X = data.drop(columns=[target])  # Todas las columnas excepto target
    y = data[target]  # Variable objetivo

    # 5. Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42)

    # 6. Preprocesamiento de datos
    # Identificar columnas numéricas y categóricas
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    # Crear un preprocesador con StandardScaler para numéricas y OneHotEncoder para categóricas
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    # 7. Crear un pipeline con el preprocesador y el modelo Lasso
    lasso_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        # alpha es el parámetro de regularización
        ('regressor', Lasso(alpha=args.alpha, random_state=42))
    ])

    # 8. Entrenar el modelo
    lasso_model.fit(X_train, y_train)

    return lasso_model, X_test, y_test


def evaluate_model(lasso_model, X_test, y_test):
    # 9. Hacer predicciones
    y_pred = lasso_model.predict(X_test)

    # 10. Evaluar el modelo usando MSE
    mse = mean_squared_error(y_test, y_pred)
    print(f"Error Cuadrático Medio (MSE): {mse:.2f}")

    # 11. Calcular la varianza de los datos objetivo
    variance = np.var(y_test)
    print(f"Varianza de los datos objetivo: {variance:.2f}")

    # 12. Comparación entre MSE y varianza
    mse_to_variance_ratio = mse / variance
    print(f"Relación MSE/Varianza: {mse_to_variance_ratio:.2f}")

    with mlflow.start_run():
        mlflow.log_metric("mse", mse)
        mlflow.log_param("alpha", args.alpha)
        mlflow.sklearn.log_model(lasso_model, "model")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", dest='input_data', type=str)
    parser.add_argument("--alpha", dest='alpha', type=float, default=0.01)
    args = parser.parse_args()

    return args


# run script
if __name__ == "__main__":
    args = parse_args()
    main(args)
