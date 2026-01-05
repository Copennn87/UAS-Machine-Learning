"""
visualization.py
Modul untuk visualisasi data dan hasil
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.tree import plot_tree

def setup_visualization():
    """Setup style untuk visualisasi"""
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12

def plot_target_distribution(y, title="Target Distribution"):
    """Plot distribusi target variable"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Count plot
    sns.countplot(x=y, ax=axes[0])
    axes[0].set_title(f'{title} - Count Plot')
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('Count')
    
    # Pie chart
    counts = y.value_counts()
    labels = [f'Class {i}' for i in counts.index]
    axes[1].pie(counts, labels=labels, autopct='%1.1f%%', startangle=90)
    axes[1].set_title(f'{title} - Pie Chart')
    
    plt.tight_layout()
    return fig

def plot_correlation_matrix(df, numerical_cols=None):
    """Plot correlation matrix heatmap"""
    if numerical_cols is None:
        numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    corr_matrix = df[numerical_cols].corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, ax=ax)
    ax.set_title('Correlation Matrix Heatmap')
    
    return fig

def plot_feature_importance(importance_df, top_n=10):
    """Plot feature importance"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    top_features = importance_df.head(top_n)
    sns.barplot(x='importance', y='feature', data=top_features, ax=ax)
    
    ax.set_title(f'Top {top_n} Feature Importance')
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    
    plt.tight_layout()
    return fig

def plot_confusion_matrix(cm, class_names=None):
    """Plot confusion matrix"""
    if class_names is None:
        class_names = ['Class 0', 'Class 1']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    
    ax.set_title('Confusion Matrix')
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    
    plt.tight_layout()
    return fig

def plot_decision_tree(model, feature_names, class_names, max_depth=3, figsize=(20, 10)):
    """Visualisasi decision tree"""
    fig, ax = plt.subplots(figsize=figsize)
    
    plot_tree(model, 
              feature_names=feature_names,
              class_names=class_names,
              filled=True,
              rounded=True,
              max_depth=max_depth,
              ax=ax)
    
    ax.set_title(f'Decision Tree Visualization (max_depth={max_depth})')
    
    return fig

def save_plot(fig, filename, dpi=300):
    """Save plot ke file"""
    fig.savefig(filename, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved: {filename}")