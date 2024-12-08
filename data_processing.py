import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(filepath):
    """Carrega o dataset."""
    df = pd.read_csv(filepath)
    return df

def preprocess_data(df):
    """Pré-processa o dataset (codificação e normalização)."""
    # Codificando variáveis categóricas
    label_encoder = LabelEncoder()
    df['mfr'] = label_encoder.fit_transform(df['mfr'])
    df['type'] = label_encoder.fit_transform(df['type'])

    # Normalizando variáveis numéricas
    scaler = StandardScaler()
    numeric_cols = ['calories', 'protein', 'fat', 'sodium', 'fiber', 'carbo', 'sugars', 'potass']
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df
