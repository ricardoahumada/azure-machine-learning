from flask import Flask, request, jsonify
import joblib
import numpy as np

# Inicializar Flask
app = Flask(__name__)

# Cargar el modelo
model = joblib.load("model.joblib")

# Definir la ruta para realizar predicciones
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Obtener datos JSON de la solicitud
        data = request.json
        print('data:', data)

        features = np.array(data["features"]).reshape(1, -1)  # Convertir a formato adecuado
        print('features:', features)
        
        # Realizar la predicción
        prediction = model.predict(features)
        
        # Devolver la predicción como respuesta JSON
        return jsonify({"prediction": int(prediction[0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Ruta de prueba
@app.route("/", methods=["GET"])
def home():
    return "API Flask funcionando correctamente"

if __name__ == "__main__":
    app.run(debug=True)