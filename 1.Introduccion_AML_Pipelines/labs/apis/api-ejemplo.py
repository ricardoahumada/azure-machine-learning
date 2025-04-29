from flask import Flask, jsonify

# Crear una instancia de Flask
app = Flask(__name__)

# Definir una ruta básica
@app.route('/')
def home():
    return "¡Bienvenido a mi aplicación Flask!"

# Definir una ruta para una API RESTful
@app.route('/api/data', methods=['GET'])
def get_data():
    # Datos de ejemplo
    data = {
        "nombre": "Flask",
        "descripcion": "Un microframework de Python para desarrollo web",
        "version": "2.3.2"
    }
    # Devolver los datos en formato JSON
    return jsonify(data)

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run(debug=True)