import matplotlib.pyplot as plt

def plot_comparison(models, mse_values, r2_values, output_path):
    """Gera e salva gráficos comparando os modelos."""
    plt.figure(figsize=(12, 6))

    # MSE
    plt.subplot(1, 2, 1)
    plt.bar(models, mse_values, color=['blue', 'orange', 'green'])
    plt.title('Comparação de MSE')
    plt.ylabel('MSE')

    # R2
    plt.subplot(1, 2, 2)
    plt.bar(models, r2_values, color=['blue', 'orange', 'green'])
    plt.title('Comparação de R²')
    plt.ylabel('R²')

    plt.tight_layout()
    plt.savefig(f"{output_path}/model_comparison.png")
    plt.show()
