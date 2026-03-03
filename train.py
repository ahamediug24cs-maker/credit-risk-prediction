"""
Reproducible training script for Credit Risk & Loan Default Prediction
Usage: python train.py
This script loads Loan_default.csv, runs preprocessing and trains the Random Forest model,
and saves artifacts to the working directory (`final_random_forest.joblib`, `scaler.joblib`, `label_encoders.joblib`).
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

CSV_PATH = 'Loan_default.csv'
MODEL_OUT = 'final_random_forest.joblib'
SCALER_OUT = 'scaler.joblib'
LE_OUT = 'label_encoders.joblib'

if __name__ == '__main__':
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"{CSV_PATH} not found. Place the dataset in the project root.")

    df = pd.read_csv(CSV_PATH)

    # Basic preprocessing (match notebook behavior)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)

    categorical_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
    le_dict = {}
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le

    # Feature engineering (minimal)
    if 'Income' in df.columns and 'CoApplicantIncome' in df.columns:
        df['Total_Income'] = df['Income'].fillna(0) + df['CoApplicantIncome'].fillna(0)
    if 'LoanAmount' in df.columns and 'Total_Income' in df.columns:
        df['Loan_to_Income_Ratio'] = df['LoanAmount'] / (df['Total_Income'] + 1)

    # Scale numeric features (exclude target if present)
    if 'Default' in df.columns:
        features = df.drop('Default', axis=1)
        y = df['Default']
    else:
        features = df
        y = None

    numeric_features = features.select_dtypes(include=[np.number]).columns.tolist()
    scaler = StandardScaler()
    features[numeric_features] = scaler.fit_transform(features[numeric_features])

    X = features

    # Train-test split
    if y is not None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    else:
        X_train, X_test, y_train, y_test = X, X, None, None

    # Train RandomForest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    if y is not None:
        rf.fit(X_train, y_train)

    # Save artifacts
    joblib.dump(rf, MODEL_OUT)
    joblib.dump(scaler, SCALER_OUT)
    joblib.dump(le_dict, LE_OUT)

    print('Training complete. Artifacts saved: ', MODEL_OUT, SCALER_OUT, LE_OUT)
