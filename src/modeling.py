'''
modeling.py
Phase C: Predictive Modeling for ATX Traffic Incidents
Focus: Logistic Regression to predict if an incident is Urgent
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from pathlib import Path


def load_data(filepath):
    print("LOADING DATA FOR PREDICTIVE MODELING...")
    return pd.read_csv(filepath)

def prepare_data_for_modeling(df):
    """
   Select features & prepare X (inputs) and y (target) for modeling.
    """
    print("\nPREPARING FEATURES FOR MODELING...")
    
    #We select features we know at the exact moment the incident is reported
    features = [
        'published_hour', 
        'published_day_of_week', 
        'is_weekend', 
        'is_rush_hour', 
        'is_night'
    ]

    X = df[features].copy()
    y = df['is_urgent'].copy()  #target variable

    return X, y, features

def train_logistic_regression_model(X, y, features, output_dir):
    """ Train a Logistic Regression model & evaluate its performance."""
    print("\n--- MACHINE LEARNING MODEL ---")

       # 1. Split data into Training Set (80%) and Test Set (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2. Scale the features so they are all on the same math scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 3. Initialize and Train the model
    print("Training Logistic Regression model (this might take a few seconds)...")
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    # 4. Make predictions on the 20% Test Set
    y_pred = model.predict(X_test_scaled)
    y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    # 5. Evaluate Performance
    print("\nClassification Report on Test Data:")
    print(sklearn.metrics.classification_report(y_test, y_pred))
    
    auc = sklearn.metrics.roc_auc_score(y_test, y_pred_prob)
    print(f"ROC-AUC Score: {auc:.3f}")
    
    # Plot ROC Curve
    plot_roc_curve(y_test, y_pred_prob, auc, output_dir)
    
    # Analyze Feature Importance
    analyze_feature_importance(model, features)

def plot_roc_curve(y_test, y_pred_prob, auc, output_dir):
    """Plot the ROC Curve."""
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_test, y_pred_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Recall)')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    plt.savefig(output_dir / '4_roc_curve.png', dpi=300)
    plt.close()
    print("\nROC Curve saved to output/visualizations!")

def analyze_feature_importance(model, features):
    """Print which features had the biggest impact on predicting urgency."""
    print("\n--- FEATURE IMPORTANCE ---")
    coefficients = model.coef_[0]
    
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance (Coefficient)': coefficients
    }).sort_values(by='Importance (Coefficient)', ascending=False)
    
    print(importance_df)
    print("\n(Positive numbers mean the feature increases the chance of being URGENT.")
    print(" Negative numbers mean the feature decreases the chance of being URGENT.)")

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    input_file = base_dir / 'data' / 'processed' / 'incidents_features.csv'
    output_dir = base_dir / 'output' / 'visualizations'
    
    if not input_file.exists():
         print(f"Error: Could not find {input_file}.")
    else:
        df = load_data(input_file)
        X, y, feature_names = prepare_data_for_modeling(df)
        train_logistic_regression_model(X, y, feature_names, output_dir)


'''
By hiding 20% of our data (the Test Set), we force the model to learn the true underlying patterns from the 80% (the Training Set). 
Then, we test it on the 20% it has never seen before. This proves how well the model can generalize to brand new, real-world data!
'''