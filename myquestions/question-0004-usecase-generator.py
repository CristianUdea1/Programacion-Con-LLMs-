import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

def generar_caso_de_uso():
    """
    Genera un caso de uso para el reto de Análisis de Sentimiento.
    Retorna una tupla: (diccionario_argumentos, objeto_resultado_esperado)
    """
    
    # --- 1. Preparación de datos de entrada ---
    data = {
        'review': [
            'Excelente producto, me encanto!', 
            'Pésimo, no funciona nada.', 
            'Lo mejor que he comprado este año.', 
            'No lo recomiendo, mala calidad.',
            'Muy satisfecho con la compra.',
            'Horrible, llego roto y tarde.',
            'Funciona perfecto y llego rápido.',
            'Una decepción total, no lo compren.',
            'Calidad premium, vale cada centavo.',
            'El peor servicio al cliente de la historia.'
        ],
        'sentiment': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    }
    df_input = pd.DataFrame(data)
    
    # --- 2. Lógica de referencia (Lo que tu función debe replicar) ---
    df = df_input.copy()
    
    # Limpieza: minúsculas y quitar puntuación básica
    df['review'] = df['review'].str.lower()
    df['review'] = df['review'].replace(r'[.,!?]', '', regex=True)
    
    # Vectorización
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['review'])
    y = df['sentiment']
    
    # División de datos (70/30)
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=0.3, 
        random_state=42
    )
    
    # Entrenamiento del modelo Naive Bayes
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    # Cálculo de métrica (Precisión/Accuracy)
    accuracy = model.score(X_test, y_test)
    
    # --- 3. Empaquetado de la tupla ---
    argumentos = {
        "df_resenas": df_input
    }
    
    # El resultado esperado contiene el vectorizador, el modelo y el score
    resultado_esperado = (vectorizer, model, accuracy)
    
    return (argumentos, resultado_esperado)

# --- Ejemplo de cómo llamar al generador y ver el contenido ---
if __name__ == "__main__":
    args, target = generar_caso_de_uso()
    
    print("Dataset de entrada (primeras filas):")
    print(args["df_resenas"].head())
    
    print("\nValores esperados:")
    print(f"- Tipo de Vectorizador: {type(target[0])}")
    print(f"- Tipo de Modelo: {type(target[1])}")
    print(f"- Accuracy esperado: {target[2]:.4f}")
