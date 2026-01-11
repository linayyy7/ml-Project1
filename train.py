#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
train.py - Training script for airline passenger satisfaction prediction
"""

import pickle
import pandas as pd
import numpy as np
import os
import warnings
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Suppress warnings
warnings.filterwarnings('ignore')

def create_output_directory():
    """Create output directory if it doesn't exist"""
    os.makedirs('output', exist_ok=True)
    print("✅ Output directory created/checked")

def load_and_clean_data(filepath):
    """Load and clean the dataset"""
    print("🛫 Loading dataset...")
    df = pd.read_csv(filepath)
    print(f"Dataset shape: {df.shape}")
    
    # Basic info
    print("\n📋 Dataset Info:")
    print("=" * 40)
    print(df.info())
    
    print("\n🔍 Missing Values:")
    print("=" * 40)
    print(df.isnull().sum())
    
    # Data cleaning
    print("\n🧹 Cleaning data...")
    df = df.copy()
    
    # Remove the first column (Unnamed: 0)
    df = df.drop(columns=['Unnamed: 0'])
    
    # Standardize column names
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    
    # Clean categorical columns
    categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)
    for c in categorical_columns:
        df[c] = df[c].str.lower().str.replace(' ', '_')
    
    # Handle missing values
    df['arrival_delay_in_minutes'] = df['arrival_delay_in_minutes'].fillna(0)
    df = df.dropna()
    
    # Convert satisfaction to binary
    df['satisfaction'] = (df['satisfaction'] == 'satisfied').astype(int)
    
    print(f"✅ Cleaned dataset shape: {df.shape}")
    return df

def split_data(df):
    """Split data into train, validation, and test sets"""
    print("\n📊 Splitting data...")
    
    # Split the data
    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
    df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)
    
    # Reset indices
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    
    # Print the sizes
    print(f"Train set: {len(df_train)} samples")
    print(f"Validation set: {len(df_val)} samples")
    print(f"Test set: {len(df_test)} samples")
    
    return df_train, df_val, df_test

def prepare_features(df_train, df_val, df_test):
    """Prepare features and target variables"""
    print("\n🔧 Preparing features...")
    
    # Define features
    numerical_features = [
        'age', 'flight_distance', 'inflight_wifi_service',
        'departure/arrival_time_convenient', 'ease_of_online_booking',
        'gate_location', 'food_and_drink', 'online_boarding', 'seat_comfort',
        'inflight_entertainment', 'on-board_service', 'leg_room_service',
        'baggage_handling', 'checkin_service', 'inflight_service',
        'cleanliness', 'departure_delay_in_minutes', 'arrival_delay_in_minutes'
    ]
    
    categorical_features = [
        'gender', 'customer_type', 'type_of_travel', 'class'
    ]
    
    # Extract target variables
    y_train = df_train['satisfaction'].values
    y_val = df_val['satisfaction'].values
    y_test = df_test['satisfaction'].values
    
    # Remove target from features
    df_train = df_train.drop(columns=['satisfaction'])
    df_val = df_val.drop(columns=['satisfaction'])
    df_test = df_test.drop(columns=['satisfaction'])
    
    # One-hot encoding
    dv = DictVectorizer(sparse=False)
    
    train_dict = df_train[categorical_features + numerical_features].to_dict(orient='records')
    X_train = dv.fit_transform(train_dict)
    
    val_dict = df_val[categorical_features + numerical_features].to_dict(orient='records')
    X_val = dv.transform(val_dict)
    
    test_dict = df_test[categorical_features + numerical_features].to_dict(orient='records')
    X_test = dv.transform(test_dict)
    
    print(f"✅ Features prepared. Training shape: {X_train.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, dv, categorical_features, numerical_features

def train_and_evaluate_models(X_train, y_train, X_val, y_val):
    """Train and evaluate multiple models"""
    print("\n🚀 Training models...")
    
    # Define models to compare
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100)
    }
    
    results = []
    trained_models = {}
    
    for name, model in models.items():
        print(f"\n  Training {name}...")
        
        # Train
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Predict
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        # Metrics
        metrics = {
            'Model': name,
            'Accuracy': accuracy_score(y_val, y_pred),
            'Precision': precision_score(y_val, y_pred),
            'Recall': recall_score(y_val, y_pred),
            'F1': f1_score(y_val, y_pred),
            'AUC': roc_auc_score(y_val, y_pred_proba),
            'model_object': model
        }
        
        results.append(metrics)
        print(f"    Accuracy: {metrics['Accuracy']:.4f}")
        print(f"    AUC: {metrics['AUC']:.4f}")
    
    # Results table
    results_df = pd.DataFrame(results).sort_values('AUC', ascending=False)
    print("\n📋 Results:")
    print("=" * 60)
    print(results_df[['Model', 'Accuracy', 'F1', 'AUC']].round(4))
    
    return results_df, trained_models

def save_best_model(results_df, trained_models, dv, categorical_features, numerical_features):
    """Save the best model to file"""
    print("\n💾 Saving best model...")
    
    # Select best model
    best_idx = results_df['AUC'].idxmax()
    best_name = results_df.loc[best_idx, 'Model']
    best_model = results_df.loc[best_idx, 'model_object']
    
    print(f"🏆 Best Model: {best_name}")
    print(f"   AUC: {results_df.loc[best_idx, 'AUC']:.4f}")
    print(f"   Accuracy: {results_df.loc[best_idx, 'Accuracy']:.4f}")
    
    # Prepare model data
    model_to_save = {
        'model': best_model,
        'vectorizer': dv,
        'numerical_features': numerical_features,
        'categorical_features': categorical_features,
        'model_name': best_name,
        'performance': results_df.loc[best_idx].to_dict()
    }
    
    # Save model
    with open('output/best_model.pkl', 'wb') as f:
        pickle.dump(model_to_save, f)
    
    print(f"✅ Model saved: output/best_model.pkl")
    
    return best_model

def test_final_model(best_model, X_test, y_test):
    """Test the final model on test set"""
    print("\n🧪 Testing final model on test set...")
    
    # Make predictions
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print("📊 Test Set Results:")
    print("=" * 40)
    print(f"Accuracy:  {accuracy:.3f}")
    print(f"F1-Score:  {f1:.3f}")
    print(f"AUC-ROC:   {auc:.3f}")
    
    return accuracy, f1, auc

def main():
    """Main training pipeline"""
    print("✈️ Airline Passenger Satisfaction Prediction - Training")
    print("=" * 50)
    
    # Create output directory
    create_output_directory()
    
    # Check if data exists
    data_path = 'Data/airline_passengers_satisfaction.csv'
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        print("Please make sure the dataset is in the Data/ folder")
        print("You can download it from:")
        print("https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction")
        return
    
    # Step 1: Load and clean data
    df = load_and_clean_data(data_path)
    
    # Step 2: Split data
    df_train, df_val, df_test = split_data(df)
    
    # Step 3: Prepare features
    X_train, X_val, X_test, y_train, y_val, y_test, dv, cat_features, num_features = prepare_features(
        df_train, df_val, df_test
    )
    
    # Step 4: Train and evaluate models
    results_df, trained_models = train_and_evaluate_models(X_train, y_train, X_val, y_val)
    
    # Step 5: Save best model
    best_model = save_best_model(results_df, trained_models, dv, cat_features, num_features)
    
    # Step 6: Test final model
    test_final_model(best_model, X_test, y_test)
    
    print("\n✅ Training completed successfully!")
    print("\n📁 To make predictions, run:")
    print("   python predict.py")

if __name__ == "__main__":
    main()