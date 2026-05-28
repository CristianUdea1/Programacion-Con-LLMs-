import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def clasificar_urgencia(df: pd.DataFrame, target_col: str) -> dict:
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    model = LogisticRegression(max_iter=300)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    errores = np.where(y_pred != y_test.values)[0]
    accuracy = (y_pred == y_test.values).mean()
    
    return {
        "accuracy": accuracy,
        "n_errores": len(errores),
        "indices_error": errores
    }