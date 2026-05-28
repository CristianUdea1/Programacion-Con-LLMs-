import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

def evaluar_finca(df: pd.DataFrame, target_col: str) -> dict:
    X = df.drop(columns=[target_col])
    y = df[target_col].to_numpy()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=7
    )
    
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    errores = np.abs(y_test - y_pred)
    
    return {
        'r2': model.score(X_test, y_test),
        'mae_promedio': errores.mean(),
        'mae_maximo': errores.max(),
        'mae_minimo': errores.min()
    }