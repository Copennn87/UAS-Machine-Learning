"""
preprocessing.py
Modul untuk handling data preprocessing
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_titanic_data(url="https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"):
    """Load dataset Titanic dari URL"""
    df = pd.read_csv(url)
    print(f"Dataset loaded. Shape: {df.shape}")
    return df

def handle_missing_values(df):
    """Handle missing values dalam dataset"""
    df_clean = df.copy()
    
    # Age - fill dengan median
    df_clean['Age'].fillna(df_clean['Age'].median(), inplace=True)
    
    # Embarked - fill dengan mode
    df_clean['Embarked'].fillna(df_clean['Embarked'].mode()[0], inplace=True)
    
    # Cabin - drop column (too many missing)
    if 'Cabin' in df_clean.columns:
        df_clean.drop('Cabin', axis=1, inplace=True)
    
    return df_clean

def feature_engineering(df):
    """Membuat feature baru"""
    df_fe = df.copy()
    
    # Family Size
    df_fe['FamilySize'] = df_fe['SibSp'] + df_fe['Parch'] + 1
    
    # Is Alone
    df_fe['IsAlone'] = (df_fe['FamilySize'] == 1).astype(int)
    
    return df_fe

def encode_categorical(df):
    """Encoding variabel kategorikal"""
    df_encoded = df.copy()
    
    # Label Encoding untuk Sex
    le = LabelEncoder()
    df_encoded['Sex_encoded'] = le.fit_transform(df_encoded['Sex'])
    
    # One-hot encoding untuk Embarked
    embarked_dummies = pd.get_dummies(df_encoded['Embarked'], prefix='Embarked')
    df_encoded = pd.concat([df_encoded, embarked_dummies], axis=1)
    
    return df_encoded

def prepare_features_target(df):
    """Menyiapkan features dan target untuk modeling"""
    # Features yang akan digunakan
    features = [
        'Pclass', 'Sex_encoded', 'Age', 'SibSp', 'Parch',
        'Fare', 'FamilySize', 'IsAlone',
        'Embarked_C', 'Embarked_Q', 'Embarked_S'
    ]
    
    # Pastikan semua features ada
    available_features = [f for f in features if f in df.columns]
    
    X = df[available_features]
    y = df['Survived']
    
    return X, y, available_features

def preprocess_pipeline(df):
    """Pipeline lengkap preprocessing"""
    print("1. Handling missing values...")
    df = handle_missing_values(df)
    
    print("2. Feature engineering...")
    df = feature_engineering(df)
    
    print("3. Encoding categorical...")
    df = encode_categorical(df)
    
    print("4. Preparing features and target...")
    X, y, features = prepare_features_target(df)
    
    print(f"Preprocessing complete. Features: {len(features)}")
    return X, y, features, df