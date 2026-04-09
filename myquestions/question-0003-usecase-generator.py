from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

def generar_caso_de_uso():
    np.random.seed(42)
    data = {
        'precio': [10, -20, 15, 100, 110, 105, 5, 12],
        'ventas': [50, 60, 55, 10, 12, 11, 70, 65]
    }
    df_input = pd.DataFrame(data)
    n_clusters = 2
    
    # Lógica interna
    df = df_input.copy()
    df['precio'] = np.abs(df['precio'])
    
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(df_scaled)
    
    # Centroides en escala original
    centroides_originales = scaler.inverse_transform(kmeans.cluster_centers_)
    
    argumentos = {"df_productos": df_input, "n_clusters": n_clusters}
    resultado_esperado = centroides_originales
    return (argumentos, resultado_esperado)
