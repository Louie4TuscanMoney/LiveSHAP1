# NBA Game Outcome Prediction - Data Collection

This repository contains scripts to collect NBA game data similar to the research paper methodology using the [nba_api](https://github.com/swar/nba_api) package.

## Research Paper Reference

Based on: "Integration of machine learning XGBoost and SHAP models for NBA game outcome prediction and quantitative analysis methodology" (PLOS ONE, 2024)

## Data Collection Overview

The scripts collect NBA game statistics for:
- **Seasons**: 2020-2021, 2021-2022, 2022-2023
- **Game periods**: First two quarters (H2), First three quarters (H3), Full game
- **Statistics**: Field goal %, 2P%, 3P%, FT%, rebounds, assists, steals, blocks, turnovers, personal fouls
- **Format**: Differences between home and away teams (home - away)

## Installation

```bash
pip install -r requirements.txt
```

## Data Collection Script

### `1datacollection/collect_nba_data_nba_api_only.py` ⭐ **MAIN SCRIPT**

Complete solution using **ONLY** the [nba_api](https://github.com/swar/nba_api) package. This script:
- Uses `LeagueGameFinder` to get all games for specified seasons (2020-21, 2021-22, 2022-23)
- Uses `BoxScoreTraditionalV2` for full game team statistics
- Uses `BoxScoreTraditionalV2` with `RangeType=2` and time-based ranges for H2 (first half) and H3 (first 3 quarters) statistics
- Calculates differences between home and away teams (matching research paper format)
- Saves each game as a separate JSON file with checkpoint/resume support

**Usage:**
```bash
cd /Users/embrace/Desktop/SHAP && python3 1datacollection/collect_nba_data_nba_api_only.py
```

**Note:** The script will resume from checkpoints if interrupted. Safe to kill and restart.

**Output:** 
- Individual game files: `Dataset/games/{GAME_ID}.json`
- Checkpoint files: `Dataset/checkpoint_{season}.json`

**Features:**
- ✅ Full game statistics (accurate and complete)
- ✅ H2 statistics (first half: 0-14400 seconds) - **Real quarter stats from API**
- ✅ H3 statistics (first 3 quarters: 0-21600 seconds) - **Real quarter stats from API**
- ✅ Checkpoint/resume functionality - safe to interrupt and restart
- ✅ Rate limiting to respect API limits
- ✅ All 61 features matching Table 1 from research paper

## Data Structure

The collected data includes:

### Full Game Statistics (game_*)
- `game_FG%`: Field goal percentage difference
- `game_2P%`: Two-point percentage difference
- `game_3P%`: Three-point percentage difference
- `game_FT%`: Free throw percentage difference
- `game_ORB`: Offensive rebounds difference
- `game_DRB`: Defensive rebounds difference
- `game_AST`: Assists difference
- `game_STL`: Steals difference
- `game_BLK`: Blocks difference
- `game_TOV`: Turnovers difference
- `game_PF`: Personal fouls difference
- `RESULT`: Home team win (1) or loss (0)

### Quarter Statistics (H2_*, H3_*)
- Same statistics as above but for:
  - **H2**: First two quarters combined
  - **H3**: First three quarters combined

## Important Notes

### Quarter Statistics Using nba_api

The script uses `BoxScoreTraditionalV2` with `RangeType=2` (time-based ranges) to get **real quarter statistics**:
- **H2 (first half)**: `range_type='2'`, `start_range='0'`, `end_range='14400'` (0-14400 seconds = Q1 + Q2)
- **H3 (first 3 quarters)**: `range_type='2'`, `start_range='0'`, `end_range='21600'` (0-21600 seconds = Q1 + Q2 + Q3)

**All statistics are accurate and come directly from the NBA API** - no estimation or aggregation needed!

### Rate Limiting

The scripts include rate limiting delays (0.6 seconds between API calls) to avoid overwhelming the NBA.com API. For large datasets, collection may take several hours.

## Data Validation

After collection, you should:
1. Remove preseason and All-Star games
2. Remove invalid entries
3. Validate data reliability (as done in the paper with ICC = 0.98)

## Next Steps

1. **Feature Engineering**: Create difference features (home - away) for all statistics
2. **Data Cleaning**: Remove outliers and invalid entries
3. **Model Training**: Use XGBoost for prediction
4. **SHAP Analysis**: Interpret model predictions

## References

- [nba_api GitHub](https://github.com/swar/nba_api)
- [Basketball Reference](https://www.basketball-reference.com/)
- Research Paper: Ouyang et al. (2024) - PLOS ONE

## License

MIT License

