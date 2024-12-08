from sklearn.metrics import mean_squared_error, r2_score

def evaluate_model(y_true, y_pred):
    """Avalia o modelo usando MSE e R²."""
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, r2
