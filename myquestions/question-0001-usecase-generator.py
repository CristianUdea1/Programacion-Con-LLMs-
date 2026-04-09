import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

def generar_caso_de_uso_0001():
    """
    Genera un caso de uso para el reto 'Predictor de Eficiencia de Combustible'.
    Retorna una tupla: (dict_argumentos, objeto_resultado_esperado)
    """
    
    # --- 1. Creación de Datos Aleatorios (Dataset Sintético) ---
    np.random.seed(42)
    n_rows = 20
    
    data = {
        'mpg': np.random.uniform(15, 45, n_rows),
        'horsepower': [100, 150, '?', 130, 95, 110, 160, 105, '?', 120, 
                       85, 90, 140, 115, 100, 105, 125, 150, 95, 110],
        'weight': np.random.uniform(2000, 5000, n_rows),
        'origin': np.random.choice(['USA', 'Japan', 'Europe'], n_rows)
    }
    df_input = pd.DataFrame(data)

    # --- 2. Lógica de Referencia (Lo que debería hacer la función del usuario) ---
    df = df_input.copy()
    
    # Limpieza
    df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
    df = df.dropna()
    
    # Feature Engineering
    df['peso_por_caballo'] = df['weight'] / df['horsepower']
    
    # Dummies
    df = pd.get_dummies(df, columns=['origin'], drop_first=True)
    
    # Split
    X = df.drop('mpg', axis=1)
    y = df['mpg']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # Escalamiento
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Modelo
    modelo = Ridge()
    modelo.fit(X_train_scaled, y_train)
    r2 = modelo.score(X_test_scaled, y_test)

    # --- 3. Construcción de la Tupla Final ---
    # El diccionario de argumentos contiene el DataFrame inicial
    argumentos = {
        "df_autos": df_input
    }
    
    # El resultado esperado es el R2 (podría ser una tupla con modelo y scaler también)
    resultado_esperado = {
        "r2_score": r2,
        "model_type": type(modelo),
        "scaler_type": type(scaler)
    }

    return (argumentos, resultado_esperado)

# --- EJEMPLO DE USO ---
inputs, targets = generar_caso_de_uso_0001()

print("--- ARGUMENTOS DE ENTRADA (Dataset) ---")
print(inputs['df_autos'].head())
print("\n--- RESULTADO ESPERADO ---")
print(f"R2 Score esperado: {targets['r2_score']:.4f}")
