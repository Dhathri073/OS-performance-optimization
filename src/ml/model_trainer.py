"""
Machine Learning Model Trainer
Trains models to predict performance metrics or identify bottlenecks.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from src.database.db_manager import load_data

MODEL_DIR = Path(__file__).parent.parent.parent / 'models'

def preprocess_data(df):
    """
    Preprocess the data for training.

    Args:
        df (pd.DataFrame): Raw performance data

    Returns:
        tuple: X, y for training
    """
    # For simplicity, predict memory_percent based on other metrics
    features = ['cpu_percent', 'cpu_freq', 'disk_percent', 'disk_used_gb', 'net_sent_mb', 'net_recv_mb']
    target = 'memory_percent'

    X = df[features].fillna(0)
    y = df[target]

    return X, y

def train_model(plot=False):
    """Train and save the ML model."""
    df = load_data(limit=10000)  # Load recent data

    if df.empty:
        print("No data available for training.")
        return

    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Model MSE: {mse:.4f}")
    print(f"Model R² Score: {r2:.4f}")

    # Save model
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / 'performance_model.pkl')
    print("Model saved.")

    if plot:
        plot_model_performance(y_test, y_pred, X_test.columns)

def plot_model_performance(y_true, y_pred, feature_names):
    """
    Plot model performance metrics.

    Args:
        y_true: True values
        y_pred: Predicted values
        feature_names: Names of features used
    """
    sns.set_style("darkgrid")
    plt.style.use('seaborn-v0_8')

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Machine Learning Model Performance', fontsize=16, fontweight='bold')

    # Predictions vs Actual
    axes[0].scatter(y_true, y_pred, alpha=0.6, color='blue', edgecolors='black')
    axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', linewidth=2)
    axes[0].set_xlabel('Actual Memory Usage (%)')
    axes[0].set_ylabel('Predicted Memory Usage (%)')
    axes[0].set_title('Predictions vs Actual Values', fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Residuals
    residuals = y_true - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.6, color='green', edgecolors='black')
    axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Predicted Values')
    axes[1].set_ylabel('Residuals')
    axes[1].set_title('Residual Plot', fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    # Feature Importance
    model = joblib.load(MODEL_DIR / 'performance_model.pkl')
    importance = model.feature_importances_
    axes[2].barh(feature_names, importance, color='skyblue', edgecolor='black')
    axes[2].set_xlabel('Importance')
    axes[2].set_title('Feature Importance', fontweight='bold')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('model_performance.png', dpi=300, bbox_inches='tight')
    print("Model performance plot saved as 'model_performance.png'")
    plt.close()

def predict_performance(features):
    """
    Predict performance using the trained model.

    Args:
        features (dict): Feature values

    Returns:
        float: Predicted value
    """
    model_path = MODEL_DIR / 'performance_model.pkl'
    if not model_path.exists():
        print("Model not found. Train the model first.")
        return None

    model = joblib.load(model_path)
    X = pd.DataFrame([features])
    prediction = model.predict(X)[0]
    return prediction

if __name__ == '__main__':
    train_model(plot=True)