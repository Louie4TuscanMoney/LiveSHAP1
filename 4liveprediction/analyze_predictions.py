#!/usr/bin/env python3
"""
Prediction Performance Analyzer
Analyzes stored predictions to evaluate model performance.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from prediction_storage import get_predictions_with_outcomes, load_predictions

def calculate_performance_metrics(df: pd.DataFrame, period_type: str = None):
    """Calculate performance metrics for predictions."""
    if df.empty:
        print("No predictions with outcomes available.")
        return None
    
    # Filter by period type if specified
    if period_type:
        df = df[df['period_type'] == period_type]
    
    if df.empty:
        print(f"No predictions found for period type: {period_type}")
        return None
    
    # Get predictions and actual outcomes
    y_pred = df['predicted_outcome'].values
    y_true = df['actual_outcome'].values
    y_proba = df['predicted_prob'].values
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # AUC (using probabilities)
    try:
        auc = roc_auc_score(y_true, y_proba)
    except ValueError:
        auc = None
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Brier score (calibration metric)
    brier_score = np.mean((y_proba - y_true) ** 2)
    
    # Calibration: Check if probabilities are well-calibrated
    # Group predictions into bins and check accuracy per bin
    bins = [0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0]
    bin_labels = ['0-30%', '30-40%', '40-50%', '50-60%', '60-70%', '70-100%']
    df['prob_bin'] = pd.cut(y_proba, bins=bins, labels=bin_labels, include_lowest=True)
    
    calibration_data = []
    for bin_label in bin_labels:
        bin_data = df[df['prob_bin'] == bin_label]
        if len(bin_data) > 0:
            calibration_data.append({
                'Probability_Range': bin_label,
                'Count': len(bin_data),
                'Predicted_Prob': bin_data['predicted_prob'].mean(),
                'Actual_Win_Rate': bin_data['actual_outcome'].mean()
            })
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc,
        'brier_score': brier_score,
        'confusion_matrix': cm,
        'calibration': pd.DataFrame(calibration_data),
        'total_predictions': len(df)
    }

def print_performance_report():
    """Print comprehensive performance report."""
    print("="*70)
    print("ML PREDICTION PERFORMANCE ANALYSIS")
    print("="*70)
    
    df = get_predictions_with_outcomes()
    
    if df.empty:
        print("\n⚠ No predictions with actual outcomes available.")
        print("  Predictions will be updated with outcomes when games finish.")
        return
    
    print(f"\nTotal Predictions with Outcomes: {len(df)}")
    print(f"Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Overall performance
    print("\n" + "="*70)
    print("OVERALL PERFORMANCE (All Periods)")
    print("="*70)
    metrics = calculate_performance_metrics(df)
    if metrics:
        print(f"\nAccuracy:  {metrics['accuracy']:.1%}")
        print(f"Precision: {metrics['precision']:.1%}")
        print(f"Recall:    {metrics['recall']:.1%}")
        print(f"F1 Score:  {metrics['f1_score']:.1%}")
        if metrics['auc']:
            print(f"AUC:       {metrics['auc']:.3f}")
        print(f"Brier Score: {metrics['brier_score']:.4f} (lower is better)")
        
        print(f"\nConfusion Matrix:")
        print(f"  True Negatives:  {metrics['confusion_matrix'][0][0]}")
        print(f"  False Positives: {metrics['confusion_matrix'][0][1]}")
        print(f"  False Negatives: {metrics['confusion_matrix'][1][0]}")
        print(f"  True Positives:  {metrics['confusion_matrix'][1][1]}")
        
        if not metrics['calibration'].empty:
            print(f"\nCalibration Analysis:")
            print(metrics['calibration'].to_string(index=False))
    
    # Performance by period type
    for period_type in ['H2', 'H3']:
        period_df = df[df['period_type'] == period_type]
        if len(period_df) > 0:
            print("\n" + "="*70)
            print(f"PERFORMANCE BY PERIOD: {period_type}")
            print("="*70)
            period_metrics = calculate_performance_metrics(df, period_type)
            if period_metrics:
                print(f"\nTotal Predictions: {period_metrics['total_predictions']}")
                print(f"Accuracy:  {period_metrics['accuracy']:.1%}")
                print(f"Precision: {period_metrics['precision']:.1%}")
                print(f"Recall:    {period_metrics['recall']:.1%}")
                print(f"F1 Score:  {period_metrics['f1_score']:.1%}")
                if period_metrics['auc']:
                    print(f"AUC:       {period_metrics['auc']:.3f}")
                print(f"Brier Score: {period_metrics['brier_score']:.4f}")
    
    # Recent predictions
    print("\n" + "="*70)
    print("RECENT PREDICTIONS")
    print("="*70)
    recent = df.tail(10).sort_values('timestamp', ascending=False)
    for _, row in recent.iterrows():
        correct = "✓" if row['predicted_outcome'] == row['actual_outcome'] else "✗"
        print(f"{correct} {row['away_team']} @ {row['home_team']} ({row['period_type']})")
        print(f"   Predicted: {row['predicted_prob_percent']:.1f}% | Actual: {'Home Win' if row['actual_outcome'] == 1 else 'Home Loss'}")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    print_performance_report()


