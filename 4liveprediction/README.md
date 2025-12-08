# Live Prediction System

This directory contains scripts for making real-time NBA game outcome predictions using trained ML models.

## Files

### `live_predictor.py`
Core prediction engine that:
- Fetches live box score data from NBA API
- Calculates features (statistical differences between home/away teams)
- Loads trained XGBoost models (H2, H3)
- Makes predictions with confidence intervals
- Displays detailed prediction output
- **Saves predictions for performance tracking**

### `monitor_live_games.py` ⭐ **MAIN SCRIPT**
Live game monitor that:
- Finds all currently live NBA games
- Monitors games for Q2 and Q3 end
- Automatically triggers ML predictions when periods end
- Displays live game status
- Tracks prediction history per game
- **Updates actual outcomes when games finish**

### `prediction_storage.py` ⭐ **NEW**
Prediction storage system that:
- Saves all predictions to JSON and CSV
- Stores prediction details (probability, confidence intervals, features)
- Updates actual outcomes when games finish
- Provides functions to load and analyze predictions

### `analyze_predictions.py` ⭐ **NEW**
Performance analyzer that:
- Calculates accuracy, precision, recall, F1, AUC
- Analyzes calibration (how well probabilities match outcomes)
- Provides performance reports by period type (H2/H3)
- Shows recent predictions and their accuracy

## Usage

### Monitor Live Games and Auto-Predict

```bash
cd /Users/embrace/Desktop/SHAP
python3 4liveprediction/monitor_live_games.py
```

**What it does:**
1. Finds all live NBA games
2. Monitors each game's period and clock
3. When Q2 ends (period changes from 2→3): Triggers H2 prediction
4. When Q3 ends (period changes from 3→4): Triggers H3 prediction
5. **Saves predictions automatically** to `4liveprediction/predictions/`
6. **Updates actual outcomes** when games finish
7. Displays live games every 30 seconds
8. Shows prediction results with confidence intervals

### Analyze Prediction Performance

```bash
python3 4liveprediction/analyze_predictions.py
```

**Output includes:**
- Overall accuracy, precision, recall, F1, AUC
- Performance by period type (H2 vs H3)
- Calibration analysis (probability vs actual win rate)
- Confusion matrix
- Recent predictions and their accuracy

## Prediction Storage

### Storage Location
- **JSON**: `4liveprediction/predictions/predictions.json`
- **CSV**: `4liveprediction/predictions/predictions.csv`

### Stored Data
Each prediction includes:
- `timestamp`: When prediction was made
- `game_id`: NBA game ID
- `period_type`: 'H2' or 'H3'
- `predicted_prob`: Predicted probability (0-1)
- `predicted_prob_percent`: Predicted probability (0-100%)
- `ci_lower`, `ci_upper`: 95% confidence interval
- `home_team`, `away_team`: Team names
- `home_score_at_prediction`, `away_score_at_prediction`: Score when prediction made
- `predicted_outcome`: Binary prediction (1 = home win, 0 = home loss)
- `actual_outcome`: Actual result (1 = home win, 0 = home loss, None = not yet known)
- `features`: All feature values used for prediction

### Automatic Outcome Updates
When games finish, the system automatically:
1. Determines actual outcome (home win/loss)
2. Updates all predictions for that game
3. Enables performance analysis

## Performance Tracking

### Metrics Calculated
- **Accuracy**: % of correct predictions
- **Precision**: % of predicted wins that actually won
- **Recall**: % of actual wins that were predicted
- **F1 Score**: Harmonic mean of precision and recall
- **AUC**: Area under ROC curve (discrimination ability)
- **Brier Score**: Calibration metric (lower is better)

### Calibration Analysis
Groups predictions by probability ranges and compares:
- Predicted probability vs actual win rate
- Shows if model is well-calibrated (e.g., 70% predictions should win ~70% of the time)

## Requirements

1. **Trained Models**: Must have `xgb_model_H2.pkl` and `xgb_model_H3.pkl` in `3modeltraining/models/`
   - Run `python3 3modeltraining/train_models.py` first if models don't exist

2. **Prepared Data**: Must have `nba_prepared_data.csv` in `2datapreparation/`
   - Run `python3 2datapreparation/prepare_data.py` first if data doesn't exist

3. **Python Packages**: See `requirements.txt` in project root

## How It Works

### Period Detection

The script detects period ends by:
1. **Period Change Detection**: Monitors when `period` field increases
   - Q2 ends: period changes from 2 → 3
   - Q3 ends: period changes from 3 → 4

2. **Clock Validation**: Confirms period end by checking game clock
   - Clock at `0:00` or empty indicates period end

3. **One-Time Prediction**: Each period (H2/H3) is predicted only once per game
   - Tracks prediction status to avoid duplicates

### Feature Calculation

When a period ends, the script:
1. Fetches live box score for the time range:
   - H2: 0-14400 seconds (first 2 quarters)
   - H3: 0-21600 seconds (first 3 quarters)

2. Calculates 11 features (home - away differences):
   - FG%, 2P%, 3P%, FT%
   - ORB, DRB, AST, STL, BLK, TOV, PF

3. Formats features to match training data format

4. Makes prediction using trained XGBoost model

5. **Saves prediction** to storage system

### Prediction Output

Each prediction includes:
- **Win Probability**: Home team's chance to win (0-100%)
- **95% Confidence Interval**: Range of uncertainty
- **Model Performance**: AUC and Accuracy for the period
- **Interpretation**: What the prediction means
- **Risk Assessment**: Analysis of confidence interval

## Troubleshooting

### "Model not found" Error
```bash
# Train models first
python3 3modeltraining/train_models.py
```

### "No live games found"
- Check if NBA games are currently in progress
- Verify internet connection
- Check NBA API endpoint availability

### Predictions not triggering
- Ensure games are actually live (not scheduled/final)
- Check that period is actually changing (Q2→Q3 or Q3→Q4)
- Verify game clock shows period end (0:00)

### Performance Analysis Shows No Data
- Wait for games to finish (outcomes updated automatically)
- Check `4liveprediction/predictions/predictions.csv` for stored predictions
- Ensure games have finished and outcomes were updated

## Notes

- **Real-time**: Script runs continuously until stopped (Ctrl+C)
- **Multiple Games**: Handles multiple live games simultaneously
- **Resilient**: Continues monitoring even if one game has errors
- **Clean Exit**: Properly handles Ctrl+C interruption
- **Performance Tracking**: All predictions automatically saved for analysis
