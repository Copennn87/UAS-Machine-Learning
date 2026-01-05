"""
model.py
Modul untuk modeling Decision Tree
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix

def train_decision_tree(X_train, y_train, max_depth=None, criterion='gini', random_state=42):
    """Train Decision Tree model"""
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        criterion=criterion,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0)
    }
    
    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    return metrics, report, cm, y_pred

def hyperparameter_tuning(X_train, y_train, X_test, y_test, depth_options=None):
    """Tuning hyperparameter max_depth"""
    if depth_options is None:
        depth_options = [3, 5, 7, 10, 15, None]
    
    results = []
    
    for depth in depth_options:
        model = train_decision_tree(X_train, y_train, max_depth=depth)
        metrics, _, _, y_pred = evaluate_model(model, X_test, y_test)
        
        results.append({
            'max_depth': depth,
            'model': model,
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'n_leaves': model.get_n_leaves(),
            'y_pred': y_pred
        })
        
        # Print progress
        depth_str = "None" if depth is None else str(depth)
        print(f"  Depth: {depth_str:>4} | "
              f"Acc: {metrics['accuracy']:.4f} | "
              f"F1: {metrics['f1']:.4f} | "
              f"Leaves: {model.get_n_leaves():3}")
    
    results_df = pd.DataFrame(results)
    
    # Pilih model terbaik berdasarkan F1-score
    if results_df['f1'].max() > 0:
        best_idx = results_df['f1'].idxmax()
    else:
        best_idx = results_df['accuracy'].idxmax()
    
    best_result = results_df.loc[best_idx]
    
    return results_df, best_result

def get_feature_importance(model, feature_names):
    """Get feature importance dari model"""
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return importance_df