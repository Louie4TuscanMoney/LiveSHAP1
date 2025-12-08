# Data Preparation

This folder contains scripts for preparing NBA game data following the methodology from the research paper (Research.md, lines 158-176).

## Script: `prepare_data.py`

This script performs the complete data preparation pipeline:

### Steps:

1. **Load Data**: Loads all JSON game files from `Dataset/games/`
2. **Remove Outliers**: Removes preseason games, All-Star games, and invalid entries
3. **Feature Selection**: Removes highly correlated features:
   - Removes: FG, FGA, 2P, 2PA, 3P, 3PA, FT, FTA, TRB
   - Keeps: FG%, 2P%, 3P%, FT%, ORB, DRB, AST, STL, BLK, TOV, PF
4. **Correlation Analysis**: Creates correlation heatmaps for game, H2, and H3 periods
5. **Descriptive Statistics**: Generates descriptive statistics tables (like Table 2)
6. **Logistic Regression**: Performs logistic regression analysis (like Tables 3-5)

### Usage:

```bash
cd /Users/embrace/Desktop/SHAP && python3 2datapreparation/prepare_data.py
```

### Output Files:

- `nba_prepared_data.csv`: Final prepared dataset
- `correlation_heatmap_game.png`: Correlation heatmap for full game
- `correlation_heatmap_H2.png`: Correlation heatmap for first half
- `correlation_heatmap_H3.png`: Correlation heatmap for first 3 quarters
- `descriptive_stats_game.csv`: Descriptive statistics for full game
- `descriptive_stats_H2.csv`: Descriptive statistics for first half
- `descriptive_stats_H3.csv`: Descriptive statistics for first 3 quarters
- `logistic_regression_game.csv`: Logistic regression results for full game
- `logistic_regression_H2.csv`: Logistic regression results for first half
- `logistic_regression_H3.csv`: Logistic regression results for first 3 quarters
- `preparation_summary.txt`: Summary of the preparation process

### Methodology:

Follows the research paper methodology:
- Feature selection based on correlation analysis (removes features with correlation ≥ 0.8)
- Keeps percentage-based features and removes makes/attempts
- Removes total rebounds (TRB) since it's redundant with ORB + DRB
- Performs statistical analysis to validate feature significance


