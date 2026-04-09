from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np
def generar_caso_de_uso_0002():
    np.random.seed(7)
    # Generamos 50 registros de 4 sensores
    data = np.random.normal(0, 1, (50, 4))
    df_input = pd.DataFrame(data, columns=['s1', 's2', 's3', 's4'])
    # Metemos un outlier manual
    df_input.iloc[0] = [10, 10, 10, 10]
    
    # Lógica interna
    df_norm = (df_input - df_input.mean()) / df_input.std()
    pca = PCA(n_components=2)
    componentes = pca.fit_transform(df_norm)
    
    iso = IsolationForest(contamination=0.1, random_state=7)
    anomalias = iso.fit_predict(componentes)
    
    df_esperado = df_input.copy()
    df_esperado['es_anomalia'] = anomalias
    
    argumentos = {"df_sensores": df_input}
    resultado_esperado = df_esperado
    return (argumentos, resultado_esperado)
