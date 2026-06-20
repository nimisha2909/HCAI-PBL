import pandas as pd
import numpy as np
from palmerpenguins import load_penguins
from sklearn.model_selection import train_test_split

def load_data():
    df = load_penguins().dropna()

    # Encode target
    df['species'] = df['species'].astype('category')
    species_labels = dict(enumerate(df['species'].cat.categories))  # {0: Adelie, 1: Chinstrap, 2: Gentoo}
    df['species_code'] = df['species'].cat.codes

    # Features to use
    feature_cols = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g',
                    'island', 'sex']

    df_model = df[feature_cols + ['species_code', 'species']].copy()

    # One-hot encode categorical columns
    df_encoded = pd.get_dummies(df_model, columns=['island', 'sex'], drop_first=False)

    X = df_encoded.drop(columns=['species_code', 'species']).values
    y = df_encoded['species_code'].values
    feature_names = df_encoded.drop(columns=['species_code', 'species']).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test, feature_names, species_labels

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, feature_names, species_labels = load_data()
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("Features:", feature_names)
    print("Classes:", species_labels)