from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

def train_models(X_train, y_train):
    """Treina os modelos de regressão."""
    # Modelos
    dt_model = DecisionTreeRegressor(random_state=42)
    lr_model = LinearRegression()
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Treinamento
    dt_model.fit(X_train, y_train)
    lr_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)

    return dt_model, lr_model, rf_model
