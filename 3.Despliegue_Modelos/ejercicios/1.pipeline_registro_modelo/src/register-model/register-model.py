
import pandas as pd
import numpy as np
import json

import argparse
from pathlib import Path

import mlflow

from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.identity import DefaultAzureCredential


def main(args):
    mlflow.autolog()

    metrics = read_metrics(args.metrics_input)
    register_model(args.model_input, metrics, args.model_name)


def read_metrics(data_path):
    # Leer las métricas desde el archivo
    with open(data_path, "r") as f:
        metrics = json.load(f)

    print(f"Métricas leídas:", metrics)

    return metrics


def register_model(model_path, metrics, model_name):
    # Inicializar el cliente de Azure ML
    ml_client = MLClient.from_config(credential=DefaultAzureCredential())

    mse = metrics['mse']
    mse_to_variance_ratio = metrics['mse_to_variance_ratio']
    threshold = 0.98

    # Registrar el modelo
    if mse_to_variance_ratio < threshold:
        registered_model = ml_client.models.create_or_update(
            Model(
                path=model_path,
                name=model_name,
                description=f"Modelo registrado con accuracy={mse}"
            )
        )
        print(f"Modelo {model_name} registrado con éxito.") 

    else:
        print(f"mse_to_variance_ratio ({mse_to_variance_ratio}) no supera el umbral ({threshold}). El modelo no será registrado ni desplegado.")




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_input", dest='model_input', type=str, required=True)
    parser.add_argument("--metrics_input", dest='metrics_input', type=str, required=True)
    parser.add_argument("--model_name", dest='model_name', type=str, required=True)
    args = parser.parse_args()

    return args


# run script
if __name__ == "__main__":
    args = parse_args()
    print('args:', args)
    main(args)
