# -*- coding: utf-8 -*-
"""Evaluation metrics for binary classification"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    roc_curve, 
    auc, 
    precision_recall_curve,
    ConfusionMatrixDisplay
)
import os
from datetime import datetime


def evaluate_model(model, test_dataset, class_names, save_plots=True):
    """Evaluate model and generate comprehensive metrics.
    
    Args:
        model: Trained Keras model
        test_dataset: Test dataset
        class_names: List of class names
        save_plots: Boolean to save plots
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    # Get predictions
    y_true = []
    y_pred_prob = []
    
    for x, y in test_dataset:
        y_true.extend(y.numpy())
        y_pred_prob.extend(model.predict(x))
    
    y_true = np.array(y_true)
    y_pred_prob = np.array(y_pred_prob)
    y_pred = (y_pred_prob >= 0.5).astype(int)
    
    # Calculate metrics
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names)
    
    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    # Calculate precision-recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
    
    # Create visualization directory
    if save_plots:
        os.makedirs('reports/figures', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Generate and save plots
    metrics = {
        'confusion_matrix': cm,
        'classification_report': report,
        'roc_auc': roc_auc,
        'fpr': fpr,
        'tpr': tpr,
        'precision': precision,
        'recall': recall
    }
    
    if save_plots:
        # Confusion Matrix plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred,
            display_labels=class_names,
            cmap='Blues',
            ax=ax
        )
        plt.title(f'Confusion Matrix ({timestamp})')
        plt.savefig(f'reports/figures/confusion_matrix_{timestamp}.png')
        plt.close()
        
        # ROC Curve plot
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve ({timestamp})')
        plt.legend()
        plt.savefig(f'reports/figures/roc_curve_{timestamp}.png')
        plt.close()
        
        # Precision-Recall curve
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label='Precision-Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve ({timestamp})')
        plt.legend()
        plt.savefig(f'reports/figures/precision_recall_{timestamp}.png')
        plt.close()
    
    return metrics 