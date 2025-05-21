
import pandas as pd
import argparse
from pathlib import Path

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
    parser.add_argument("--output_data", dest='output_data',type=str)
    args = parser.parse_args()

    return args


# run script
if __name__ == "__main__":
    args = parse_args()
    main(args)
