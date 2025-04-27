import argparse
import pandas as pd
import os

def main():
    # Parsear argumentos
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_data", type=str, help="Ruta de salida para el archivo CSV")
    args = parser.parse_args()

    # Crear un DataFrame de ejemplo
    df = pd.DataFrame({
        "nombre": ["Alice", "Bob", "Charlie"],
        "edad": [25, 30, 35],
        "ciudad": ["Nueva York", "Los Ángeles", "Chicago"]
    })

    # Guardar el DataFrame como CSV en la ruta de salida
    output_path = os.path.join(args.output_data, "data.csv")
    df.to_csv(output_path, index=False)
    print(f"DataFrame guardado en: {output_path}")

if __name__ == "__main__":
    main()