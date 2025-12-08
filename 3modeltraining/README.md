# Model Training

This folder contains scripts for training NBA game outcome prediction models following the methodology from the research paper (Research.md, lines 177-212).

## Script: `train_models.py`

This script performs the complete model training pipeline:

### Steps:

1. **Load Prepared Data**: Loads the prepared dataset from `2datapreparation/nba_prepared_data.csv`
2. **Group by Period**: Creates separate datasets for H2, H3, and full game periods
3. **Hyperparameter Tuning**: Uses Bayesian optimization and grid search for:
   - XGBoost
   - LightGBM
   - Random Forest
   - SVM
   - KNN
   - Logistic Regression
   - Decision Tree
4. **10-Fold Cross-Validation**: Evaluates each model using stratified 10-fold CV
5. **Performance Metrics**: Calculates AUC, F1 Score, accuracy, precision, recall
6. **SHAP Analysis**: Performs interpretability analysis on best XGBoost models

### Usage:

```bash
cd /Users/embrace/Desktop/SHAP && python3 3modeltraining/train_models.py
```

### Output Files:

- `results_H2.csv`: Performance metrics for first two quarters period (Table 7)
- `results_H3.csv`: Performance metrics for first three quarters period (Table 8)
- `results_game.csv`: Performance metrics for full game period (Table 9)
- `model_comparison_chart.png`: Visual comparison of all models (Fig 5)
- `shap_summary_H2.png`: SHAP summary plot for H2 period
- `shap_summary_H3.png`: SHAP summary plot for H3 period
- `shap_summary_game.png`: SHAP summary plot for full game
- `shap_importance_H2.csv`: Feature importance rankings (Table 10)
- `shap_importance_H3.csv`: Feature importance rankings
- `shap_importance_game.csv`: Feature importance rankings
- `models/xgb_model_H2.pkl`: Trained XGBoost model for H2
- `models/xgb_model_H3.pkl`: Trained XGBoost model for H3
- `models/xgb_model_game.pkl`: Trained XGBoost model for full game

### Methodology:

Follows the research paper methodology:
- Groups data by period (H2, H3, game) for real-time prediction
- Uses Bayesian optimization and grid search for hyperparameter tuning
- Performs 10-fold cross-validation for robust evaluation
- Evaluates using 5 metrics: AUC, F1 Score, accuracy, precision, recall
- Uses SHAP for model interpretability and feature importance analysis

### Expected Results:

Based on the research paper:
- XGBoost should perform best overall
- LightGBM should be second best
- Models should improve performance as more game data is available (H2 < H3 < game)


