#!/usr/bin/env python3
"""
Prediction Storage System
Stores ML predictions for performance tracking and analysis.
"""

import json
import csv
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

# Storage directory
STORAGE_DIR = Path('4liveprediction/predictions')
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

# Prediction files
PREDICTIONS_JSON = STORAGE_DIR / 'predictions.json'
PREDICTIONS_CSV = STORAGE_DIR / 'predictions.csv'

def save_prediction(
    game_id: str,
    period_type: str,
    predicted_prob: float,
    ci_lower: Optional[float] = None,
    ci_upper: Optional[float] = None,
    home_team: str = '',
    away_team: str = '',
    home_score: int = 0,
    away_score: int = 0,
    features: Optional[Dict] = None,
    actual_outcome: Optional[int] = None  # 1 = home win, 0 = home loss
):
    """
    Save a prediction to storage.
    
    Args:
        game_id: NBA game ID
        period_type: 'H2' or 'H3'
        predicted_prob: Predicted probability (0-1)
        ci_lower: Lower bound of confidence interval
        ci_upper: Upper bound of confidence interval
        home_team: Home team name
        away_team: Away team name
        home_score: Home team score at prediction time
        away_score: Away team score at prediction time
        features: Dictionary of features used for prediction
        actual_outcome: Actual game outcome (1 = home win, 0 = home loss, None = not yet known)
    """
    timestamp = datetime.now().isoformat()
    
    prediction = {
        'timestamp': timestamp,
        'game_id': str(game_id),
        'period_type': period_type,
        'predicted_prob': float(predicted_prob),
        'predicted_prob_percent': float(predicted_prob) * 100,
        'ci_lower': float(ci_lower) if ci_lower is not None else None,
        'ci_upper': float(ci_upper) if ci_upper is not None else None,
        'home_team': str(home_team),
        'away_team': str(away_team),
        'home_score_at_prediction': int(home_score),
        'away_score_at_prediction': int(away_score),
        'actual_outcome': actual_outcome,  # Will be updated later when game finishes
        'predicted_outcome': 1 if predicted_prob >= 0.5 else 0,  # Binary prediction
        'features': features if features else {}
    }
    
    # Save to JSON (append mode)
    try:
        if PREDICTIONS_JSON.exists():
            with open(PREDICTIONS_JSON, 'r') as f:
                predictions_list = json.load(f)
        else:
            predictions_list = []
        
        predictions_list.append(prediction)
        
        with open(PREDICTIONS_JSON, 'w') as f:
            json.dump(predictions_list, f, indent=2)
    except Exception as e:
        print(f"⚠ Error saving prediction to JSON: {e}")
    
    # Save to CSV (append mode)
    try:
        # Convert to DataFrame row
        csv_row = {
            'timestamp': prediction['timestamp'],
            'game_id': prediction['game_id'],
            'period_type': prediction['period_type'],
            'predicted_prob': prediction['predicted_prob'],
            'predicted_prob_percent': prediction['predicted_prob_percent'],
            'ci_lower': prediction['ci_lower'],
            'ci_upper': prediction['ci_upper'],
            'home_team': prediction['home_team'],
            'away_team': prediction['away_team'],
            'home_score_at_prediction': prediction['home_score_at_prediction'],
            'away_score_at_prediction': prediction['away_score_at_prediction'],
            'predicted_outcome': prediction['predicted_outcome'],
            'actual_outcome': prediction['actual_outcome']
        }
        
        # Add feature columns
        if features:
            for key, value in features.items():
                csv_row[f'feature_{key}'] = value
        
        # Append to CSV
        file_exists = PREDICTIONS_CSV.exists()
        with open(PREDICTIONS_CSV, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(csv_row)
    except Exception as e:
        print(f"⚠ Error saving prediction to CSV: {e}")

def update_actual_outcome(game_id: str, actual_outcome: int):
    """
    Update the actual outcome for a game's predictions.
    
    Args:
        game_id: NBA game ID
        actual_outcome: 1 = home win, 0 = home loss
    """
    # Update JSON
    try:
        if PREDICTIONS_JSON.exists():
            with open(PREDICTIONS_JSON, 'r') as f:
                predictions_list = json.load(f)
            
            updated = False
            for pred in predictions_list:
                if pred['game_id'] == str(game_id) and pred['actual_outcome'] is None:
                    pred['actual_outcome'] = int(actual_outcome)
                    updated = True
            
            if updated:
                with open(PREDICTIONS_JSON, 'w') as f:
                    json.dump(predictions_list, f, indent=2)
    except Exception as e:
        print(f"⚠ Error updating outcome in JSON: {e}")
    
    # Update CSV
    try:
        if PREDICTIONS_CSV.exists():
            df = pd.read_csv(PREDICTIONS_CSV)
            mask = (df['game_id'] == str(game_id)) & (df['actual_outcome'].isna())
            df.loc[mask, 'actual_outcome'] = int(actual_outcome)
            df.to_csv(PREDICTIONS_CSV, index=False)
    except Exception as e:
        print(f"⚠ Error updating outcome in CSV: {e}")

def load_predictions() -> pd.DataFrame:
    """Load all predictions as a DataFrame."""
    if PREDICTIONS_CSV.exists():
        return pd.read_csv(PREDICTIONS_CSV)
    return pd.DataFrame()

def get_predictions_with_outcomes() -> pd.DataFrame:
    """Get predictions that have actual outcomes (for performance analysis)."""
    df = load_predictions()
    if df.empty:
        return df
    return df[df['actual_outcome'].notna()]


