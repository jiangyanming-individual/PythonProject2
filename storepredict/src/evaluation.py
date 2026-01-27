import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_model(true, predicted):
    mae = mean_absolute_error(true, predicted)
    mse = mean_squared_error(true, predicted)
    rmse = np.sqrt(mse)
    r2_square = r2_score(true, predicted)
    return mae, mse, rmse, r2_square

def evaluate_all_models(models, X_test, y_test):
    results = []
    for name, model in models.items():
        print(f"正在评估 {name}...")
        y_pred = model.predict(X_test)
        mae, mse, rmse, r2 = evaluate_model(y_test, y_pred)
        results.append({
            'Model': name,
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2
        })
    
    return pd.DataFrame(results)
