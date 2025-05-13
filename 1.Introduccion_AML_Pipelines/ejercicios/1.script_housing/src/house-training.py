
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

# 1. Cargar los datos
# Suponemos que el dataset está en un archivo CSV llamado 'house_prices.csv'
data = pd.read_csv("./house_prices.csv")

# 2. Inspeccionar las columnas
print(data.columns)

# 3. Descartar columnas no significativas
# Las columnas 'date', 'street', 'city', 'statezip', 'country' no son relevantes para la predicción
columns_to_drop = ['date', 'street', 'city', 'statezip', 'country']
data = data.drop(columns=columns_to_drop)

# 4. Separar características (X) y objetivo (y)
X = data.drop(columns=['price'])  # Todas las columnas excepto 'price'
y = data['price']  # Variable objetivo

# 5. Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
    ('regressor', Lasso(alpha=1, random_state=42))  # alpha es el parámetro de regularización
])

# 8. Entrenar el modelo
lasso_model.fit(X_train, y_train)

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
