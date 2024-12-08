import os
from src.data_processing import load_data, preprocess_data
from src.model_training import train_models
from src.evaluation import evaluate_model
from src.visualization import plot_comparison
from sklearn.model_selection import train_test_split

def main():
    # Carregar e pré-processar os dados
    df = load_data('data/cereal.csv')
    df = preprocess_data(df)

    # Dividir os dados em treino e teste
    X = df[['calories', 'protein', 'fat', 'sodium', 'fiber', 'carbo', 'sugars', 'potass', 'mfr', 'type']]
    y = df['rating']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Treinar os modelos
    dt_model, lr_model, rf_model = train_models(X_train, y_train)

    # Fazer previsões
    dt_pred = dt_model.predict(X_test)
    lr_pred = lr_model.predict(X_test)
    rf_pred = rf_model.predict(X_test)

    # Avaliar os modelos
    models = ['Decision Tree', 'Linear Regression', 'Random Forest']
    mse_values = [
        mean_squared_error(y_test, dt_pred),
        mean_squared_error(y_test, lr_pred),
        mean_squared_error(y_test, rf_pred)
    ]
    r2_values = [
        r2_score(y_test, dt_pred),
        r2_score(y_test, lr_pred),
        r2_score(y_test, rf_pred)
    ]

    # Salvar e exibir os resultados
    os.makedirs('outputs/plots', exist_ok=True)
    plot_comparison(models, mse_values, r2_values, 'outputs')

if __name__ == "__main__":
    main()
