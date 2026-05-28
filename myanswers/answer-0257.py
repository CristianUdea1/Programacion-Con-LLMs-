import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score

def optimizar_zonas_davies(X: np.ndarray, k_min: int = 2, k_max: int = 10) -> tuple:
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    scores = []
    for k in range(k_min, k_max + 1):
        modelo = KMeans(n_clusters=k, random_state=42)
        etiquetas = modelo.fit_predict(X_scaled)
        score = davies_bouldin_score(X_scaled, etiquetas)
        scores.append(score)
    
    scores = np.array(scores)
    indice_mejor = np.argmin(scores)
    k_optimo = range(k_min, k_max + 1)[indice_mejor]
    score_minimo = round(scores[indice_mejor], 4)
    
    return (k_optimo, score_minimo)