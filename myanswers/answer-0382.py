import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer
from sklearn.decomposition import PCA

def procesar_datos_financieros(df: pd.DataFrame, n_componentes: int) -> np.ndarray:
    df_limpio = df.copy()
    df_limpio = df_limpio.fillna(df_limpio.mean())
    pt = PowerTransformer()
    datos_transformados = pt.fit_transform(df_limpio)
    pca = PCA(n_components=n_componentes)
    datos_finales = pca.fit_transform(datos_transformados)
    return datos_finales