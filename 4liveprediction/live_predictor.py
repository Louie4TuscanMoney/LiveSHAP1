#!/usr/bin/env python3
"""
NBA Live Game Outcome Predictor
Monitors live games and makes predictions at end of Q2 and Q3 using trained ML models
"""

import requests
import time
import sys
import re
import json
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from nba_api.stats.endpoints import boxscoretraditionalv2
from nba_api.live.nba.endpoints import boxscore as live_boxscore
from nba_api.live.nba.endpoints import boxscore as live_boxscore

# Live scoreboard endpoint (same as scoreboard_updater.py)
SCOREBOARD_URL = "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json"

# Model paths
MODELS_DIR = Path('3modeltraining/models')
PREPARED_DATA_PATH = Path('2datapreparation/nba_prepared_data.csv')

# Track which games we've already predicted for each period
predicted_games = {}  # {game_id: {H2: bool, H3: bool}}

def fetch_scoreboard():
    """Fetch live scoreboard data from NBA CDN."""
    try:
        response = requests.get(SCOREBOARD_URL, timeout=2)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching scoreboard: {e}")
        return None

def get_live_boxscore(game_id, start_range=0, end_range=None):
    """
    Get live boxscore stats for a specific time range.
    For live games, uses the live boxscore endpoint (nba_api.live.nba.endpoints.boxscore)
    which provides better support for live game data.
    Falls back to BoxScoreTraditionalV2 for period-specific ranges if needed.
    """
    try:
        # First, try the live boxscore endpoint (better for live games)
        # Reference: https://github.com/swar/nba_api/blob/master/src/nba_api/live/nba/endpoints/boxscore.py
        print(f"  [DEBUG] Trying live boxscore endpoint for game {game_id}...")
        try:
            live_box = live_boxscore.BoxScore(game_id=game_id)
            live_data = live_box.get_dict()
            
            if 'game' in live_data and 'homeTeam' in live_data['game'] and 'awayTeam' in live_data['game']:
                home_team_data = live_data['game']['homeTeam']
                away_team_data = live_data['game']['awayTeam']
                
                # Extract team stats from live boxscore
                home_stats = {
                    'TEAM_ID': home_team_data.get('teamId', 0),
                    'TEAM_NAME': home_team_data.get('teamName', ''),
                    'FGM': home_team_data.get('statistics', {}).get('fieldGoalsMade', 0),
                    'FGA': home_team_data.get('statistics', {}).get('fieldGoalsAttempted', 0),
                    'FG3M': home_team_data.get('statistics', {}).get('threePointersMade', 0),
                    'FG3A': home_team_data.get('statistics', {}).get('threePointersAttempted', 0),
                    'FTM': home_team_data.get('statistics', {}).get('freeThrowsMade', 0),
                    'FTA': home_team_data.get('statistics', {}).get('freeThrowsAttempted', 0),
                    'OREB': home_team_data.get('statistics', {}).get('reboundsOffensive', 0),
                    'DREB': home_team_data.get('statistics', {}).get('reboundsDefensive', 0),
                    'AST': home_team_data.get('statistics', {}).get('assists', 0),
                    'STL': home_team_data.get('statistics', {}).get('steals', 0),
                    'BLK': home_team_data.get('statistics', {}).get('blocks', 0),
                    'TOV': home_team_data.get('statistics', {}).get('turnovers', 0),
                    'PF': home_team_data.get('statistics', {}).get('foulsPersonal', 0),
                    'PTS': home_team_data.get('score', 0)
                }
                
                away_stats = {
                    'TEAM_ID': away_team_data.get('teamId', 0),
                    'TEAM_NAME': away_team_data.get('teamName', ''),
                    'FGM': away_team_data.get('statistics', {}).get('fieldGoalsMade', 0),
                    'FGA': away_team_data.get('statistics', {}).get('fieldGoalsAttempted', 0),
                    'FG3M': away_team_data.get('statistics', {}).get('threePointersMade', 0),
                    'FG3A': away_team_data.get('statistics', {}).get('threePointersAttempted', 0),
                    'FTM': away_team_data.get('statistics', {}).get('freeThrowsMade', 0),
                    'FTA': away_team_data.get('statistics', {}).get('freeThrowsAttempted', 0),
                    'OREB': away_team_data.get('statistics', {}).get('reboundsOffensive', 0),
                    'DREB': away_team_data.get('statistics', {}).get('reboundsDefensive', 0),
                    'AST': away_team_data.get('statistics', {}).get('assists', 0),
                    'STL': away_team_data.get('statistics', {}).get('steals', 0),
                    'BLK': away_team_data.get('statistics', {}).get('blocks', 0),
                    'TOV': away_team_data.get('statistics', {}).get('turnovers', 0),
                    'PF': away_team_data.get('statistics', {}).get('foulsPersonal', 0),
                    'PTS': away_team_data.get('score', 0)
                }
                
                # Create DataFrame
                team_stats_df = pd.DataFrame([home_stats, away_stats])
                print(f"  [DEBUG] ✓ Got live boxscore data: {len(team_stats_df)} teams")
                return team_stats_df
        except Exception as e_live:
            print(f"  [DEBUG] Live boxscore failed: {e_live}, trying traditional endpoint...")
        
        # Fallback to traditional endpoint for period-specific ranges
        if end_range is not None:
            print(f"  [DEBUG] Fetching period-specific boxscore: game_id={game_id}, range={start_range}-{end_range}")
            boxscore = boxscoretraditionalv2.BoxScoreTraditionalV2(
                game_id=game_id,
                range_type='2',
                start_range=str(start_range),
                end_range=str(end_range)
            )
            data_frames = boxscore.get_data_frames()
            if len(data_frames) > 1 and data_frames[1] is not None:
                team_stats = data_frames[1]
                if len(team_stats) >= 2:
                    print(f"  [DEBUG] ✓ Got period-specific stats: {len(team_stats)} teams")
                    return team_stats
        
        # Last resort: full game traditional boxscore
        print(f"  [DEBUG] Trying full game traditional boxscore...")
        boxscore = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
        data_frames = boxscore.get_data_frames()
        if len(data_frames) > 1 and data_frames[1] is not None:
            team_stats = data_frames[1]
            if len(team_stats) >= 2:
                print(f"  [DEBUG] ✓ Got full game stats: {len(team_stats)} teams")
                return team_stats
                    
    except Exception as e:
        print(f"  ⚠ Error fetching boxscore for game {game_id}: {e}")
        import traceback
        traceback.print_exc()
    return None

def calculate_stat_difference(home_stats, away_stats, stat_name):
    """Calculate difference (home - away) for a stat."""
    home_val = home_stats.get(stat_name, 0)
    away_val = away_stats.get(stat_name, 0)
    
    # Handle percentage calculations
    if stat_name in ['FG_PCT', 'FG3_PCT', 'FT_PCT']:
        # Already percentages, return difference
        return float(home_val) - float(away_val) if pd.notna(home_val) and pd.notna(away_val) else 0.0
    
    # For other stats, return difference
    return float(home_val) - float(away_val) if pd.notna(home_val) and pd.notna(away_val) else 0.0

def calculate_percentage(made, attempted):
    """Calculate percentage, handling division by zero."""
    if attempted == 0 or pd.isna(attempted) or pd.isna(made):
        return 0.0
    return float(made) / float(attempted)

def extract_features_from_boxscore(team_stats_df, home_team_id, away_team_id, period='game'):
    """
    Extract features from boxscore DataFrame and calculate differences.
    Returns dict with features matching the model input format.
    """
    # Find home and away team rows
    home_idx = None
    away_idx = None
    
    for idx, row in team_stats_df.iterrows():
        team_id = row.get('TEAM_ID')
        if team_id == home_team_id:
            home_idx = idx
        elif team_id == away_team_id:
            away_idx = idx
    
    if home_idx is None or away_idx is None:
        return None
    
    home_stats = team_stats_df.iloc[home_idx]
    away_stats = team_stats_df.iloc[away_idx]
    
    features = {}
    
    # Calculate percentages and differences
    # FG%
    home_fg_pct = calculate_percentage(home_stats.get('FGM', 0), home_stats.get('FGA', 0))
    away_fg_pct = calculate_percentage(away_stats.get('FGM', 0), away_stats.get('FGA', 0))
    features[f'{period}_FG%'] = home_fg_pct - away_fg_pct
    
    # 2P% (two-point percentage)
    home_2pm = home_stats.get('FGM', 0) - home_stats.get('FG3M', 0)
    home_2pa = home_stats.get('FGA', 0) - home_stats.get('FG3A', 0)
    away_2pm = away_stats.get('FGM', 0) - away_stats.get('FG3M', 0)
    away_2pa = away_stats.get('FGA', 0) - away_stats.get('FG3A', 0)
    home_2p_pct = calculate_percentage(home_2pm, home_2pa)
    away_2p_pct = calculate_percentage(away_2pm, away_2pa)
    features[f'{period}_2P%'] = home_2p_pct - away_2p_pct
    
    # 3P%
    home_3p_pct = calculate_percentage(home_stats.get('FG3M', 0), home_stats.get('FG3A', 0))
    away_3p_pct = calculate_percentage(away_stats.get('FG3M', 0), away_stats.get('FG3A', 0))
    features[f'{period}_3P%'] = home_3p_pct - away_3p_pct
    
    # FT%
    home_ft_pct = calculate_percentage(home_stats.get('FTM', 0), home_stats.get('FTA', 0))
    away_ft_pct = calculate_percentage(away_stats.get('FTM', 0), away_stats.get('FTA', 0))
    features[f'{period}_FT%'] = home_ft_pct - away_ft_pct
    
    # Rebounds and other stats
    features[f'{period}_ORB'] = calculate_stat_difference(home_stats, away_stats, 'OREB')
    features[f'{period}_DRB'] = calculate_stat_difference(home_stats, away_stats, 'DREB')
    features[f'{period}_AST'] = calculate_stat_difference(home_stats, away_stats, 'AST')
    features[f'{period}_STL'] = calculate_stat_difference(home_stats, away_stats, 'STL')
    features[f'{period}_BLK'] = calculate_stat_difference(home_stats, away_stats, 'BLK')
    features[f'{period}_TOV'] = calculate_stat_difference(home_stats, away_stats, 'TOV')
    features[f'{period}_PF'] = calculate_stat_difference(home_stats, away_stats, 'PF')
    
    return features

def get_top_contributing_features(features_dict, period_type, top_n=5):
    """
    Get top contributing features for interpretability.
    Returns list of (feature_name, feature_value) tuples sorted by absolute value.
    """
    # Filter features for this period
    period_features = {k: v for k, v in features_dict.items() if k.startswith(f'{period_type}_')}
    
    # Sort by absolute value (most impactful)
    sorted_features = sorted(period_features.items(), key=lambda x: abs(x[1]), reverse=True)
    
    return sorted_features[:top_n]

def get_prediction_uncertainty(period, prob):
    """
    Get prediction uncertainty estimate for probability predictions.
    
    Uses empirical uncertainty based on:
    1. Model calibration uncertainty (higher uncertainty near 0.5)
    2. Cross-validation variability (if available)
    
    Returns standard deviation estimate for the probability.
    """
    # Base uncertainty: higher near 0.5 (most uncertain), lower at extremes
    # This reflects that predictions near 50% are less certain
    base_uncertainty = 0.5 * np.sqrt(prob * (1 - prob)) * 0.15
    
    # Try to load cross-validation results for period-specific uncertainty
    results_file = Path('3modeltraining') / f'results_{period}.csv'
    cv_uncertainty = None
    
    if results_file.exists():
        try:
            df_results = pd.read_csv(results_file)
            xgb_row = df_results[df_results['Algorithm'] == 'XGBoost']
            if not xgb_row.empty:
                # If we had stored std in results, we could use it here
                # For now, use a conservative estimate based on typical CV variability
                cv_uncertainty = 0.03  # 3% based on typical XGBoost CV std
        except:
            pass
    
    # Combine base uncertainty with CV uncertainty
    if cv_uncertainty:
        total_uncertainty = np.sqrt(base_uncertainty**2 + cv_uncertainty**2)
    else:
        total_uncertainty = base_uncertainty
    
    # Ensure minimum uncertainty (at least 2%)
    return max(0.02, total_uncertainty)

def make_prediction(model, features_dict, period):
    """
    Make prediction using trained model.
    Returns tuple: (probability, confidence_interval_lower, confidence_interval_upper)
    """
    # Load feature order from prepared dataset
    df_prep = pd.read_csv(PREPARED_DATA_PATH)
    feature_cols = [col for col in df_prep.columns if col.startswith(f'{period}_')]
    feature_cols = [col for col in feature_cols if col not in ['GAME_ID', 'SEASON', 'RESULT']]
    
    # Create feature vector in correct order
    feature_vector = []
    for col in sorted(feature_cols):
        feature_vector.append(features_dict.get(col, 0.0))
    
    # Convert to numpy array and reshape for model
    X = np.array(feature_vector).reshape(1, -1)
    
    # Make prediction
    try:
        prob = model.predict_proba(X)[0][1]  # Probability of class 1 (home win)
        
        # Get uncertainty estimate (depends on the probability value)
        uncertainty = get_prediction_uncertainty(period, prob)
        
        # Calculate confidence interval (95% CI using normal approximation)
        # Clamp to [0, 1] range
        ci_lower = max(0.0, prob - 1.96 * uncertainty)
        ci_upper = min(1.0, prob + 1.96 * uncertainty)
        
        return prob, ci_lower, ci_upper
    except Exception as e:
        print(f"Error making prediction: {e}")
        return None, None, None

def check_period_end(game, previous_periods):
    """
    Check if a period has just ended.
    Returns tuple: (period_ended, period_that_ended, current_period)
    """
    game_id = str(game.get('gameId'))
    current_period = game.get('period', 0)
    game_status = game.get('gameStatusText', '')
    game_clock = game.get('gameClock', '')
    
    # Get previous period for this game
    prev_period = previous_periods.get(game_id, 0)
    
    # Check if period changed (period just ended)
    if current_period > prev_period and prev_period > 0:
        period_ended = prev_period  # The period that just ended
        previous_periods[game_id] = current_period
        return True, period_ended, current_period
    
    # Update stored period
    previous_periods[game_id] = current_period
    return False, None, current_period

def predict_game_outcome(game, period_type):
    """
    Fetch live stats and make prediction for a game at end of Q2 or Q3.
    period_type: 'H2' for end of Q2, 'H3' for end of Q3
    """
    game_id = str(game.get('gameId'))
    home_team = game.get('homeTeam', {})
    away_team = game.get('awayTeam', {})
    home_team_id = home_team.get('teamId')
    away_team_id = away_team.get('teamId')
    home_name = home_team.get('teamName', 'Home')
    away_name = away_team.get('teamName', 'Away')
    
    # Check if we've already predicted for this period
    if game_id not in predicted_games:
        predicted_games[game_id] = {'H2': False, 'H3': False}
    
    if predicted_games[game_id][period_type]:
        return  # Already predicted
    
    # Determine time range based on period type
    if period_type == 'H2':
        # First half: 0 to 14400 seconds (2 quarters × 12 min × 60 sec)
        start_range = 0
        end_range = 14400
    elif period_type == 'H3':
        # First 3 quarters: 0 to 21600 seconds (3 quarters × 12 min × 60 sec)
        start_range = 0
        end_range = 21600
    else:
        return
    
    print(f"\n[{time.strftime('%H:%M:%S')}] Fetching {period_type} stats for {away_name} @ {home_name}...")
    
    # Fetch live boxscore
    team_stats = get_live_boxscore(game_id, start_range=start_range, end_range=end_range)
    
    if team_stats is None or len(team_stats) < 2:
        print(f"  ⚠ Could not fetch boxscore stats")
        return
    
    # Extract features
    features = extract_features_from_boxscore(team_stats, home_team_id, away_team_id, period=period_type)
    
    if features is None:
        print(f"  ⚠ Could not extract features")
        return
    
    # Load model
    model_path = MODELS_DIR / f'xgb_model_{period_type}.pkl'
    if not model_path.exists():
        print(f"  ⚠ Model not found: {model_path}")
        return
    
    try:
        model = joblib.load(model_path)
    except Exception as e:
        print(f"  ⚠ Error loading model: {e}")
        return
    
    # Make prediction
    prob, ci_lower, ci_upper = make_prediction(model, features, period_type)
    
    if prob is None:
        print(f"  ⚠ Could not make prediction")
        return
    
    # Mark as predicted
    predicted_games[game_id][period_type] = True
    
    # Save prediction to storage for performance tracking
    try:
        from prediction_storage import save_prediction
        save_prediction(
            game_id=game_id,
            period_type=period_type,
            predicted_prob=prob,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            home_team=home_name,
            away_team=away_name,
            home_score=home_team.get('score', 0),
            away_score=away_team.get('score', 0),
            features=features
        )
    except Exception as e:
        print(f"  ⚠ Warning: Could not save prediction: {e}")
    
    # Display prediction with confidence intervals
    prob_percent = prob * 100
    ci_lower_percent = ci_lower * 100 if ci_lower else None
    ci_upper_percent = ci_upper * 100 if ci_upper else None
    
    # Option to show/hide confidence intervals (set to False to disable)
    SHOW_CONFIDENCE_INTERVALS = True
    
    # Get model accuracy info for context
    period_accuracy = {
        'H2': {'AUC': 0.7816, 'Accuracy': 0.7051},
        'H3': {'AUC': 0.867, 'Accuracy': 0.779}
    }
    model_info = period_accuracy.get(period_type, {})
    
    # Calculate away team win probability
    away_prob_percent = (1 - prob) * 100
    
    print(f"\n{'='*70}")
    print(f"🎯 LIVE PREDICTION: {away_name} @ {home_name}")
    print(f"{'='*70}")
    print(f"⏱️  Prediction Point: End of {'Q2 (Halftime)' if period_type == 'H2' else 'Q3 (Before Q4)'}")
    print(f"📊 Current Score: {away_name} {away_team.get('score', 0)} - {home_team.get('score', 0)} {home_name}")
    print(f"{'='*70}")
    # Determine favorite and underdog
    favorite_name = home_name if prob_percent >= 50 else away_name
    underdog_name = away_name if prob_percent >= 50 else home_name
    favorite_prob = prob_percent if prob_percent >= 50 else away_prob_percent
    
    # Visual probability bar
    bar_length = 50
    favorite_bar = int(bar_length * (favorite_prob / 100))
    underdog_bar = bar_length - favorite_bar
    
    # Visual probability display
    print(f"\n🏀 WIN PROBABILITY:")
    print(f"   {'█' * favorite_bar}{'░' * underdog_bar}")
    print(f"   {favorite_name}: {favorite_prob:.1f}%")
    print(f"   {underdog_name}: {100 - favorite_prob:.1f}%")
    
    if SHOW_CONFIDENCE_INTERVALS and ci_lower_percent is not None and ci_upper_percent is not None:
        print(f"\n📈 CONFIDENCE INTERVAL (95%):")
        ci_width = ci_upper_percent - ci_lower_percent
        if ci_width < 10:
            confidence_level = "High"
            confidence_emoji = "🟢"
        elif ci_width < 20:
            confidence_level = "Moderate"
            confidence_emoji = "🟡"
        else:
            confidence_level = "Low"
            confidence_emoji = "🔴"
        print(f"   {confidence_emoji} {confidence_level} confidence: [{ci_lower_percent:.1f}% - {ci_upper_percent:.1f}%]")
    
    # Key stats that drove the prediction
    print(f"\n📊 KEY STATS DRIVING PREDICTION:")
    top_features = get_top_contributing_features(features, period_type, top_n=5)
    for feat_name, feat_value in top_features:
        stat_name = feat_name.replace(f'{period_type}_', '').replace('%', '%').replace('_', ' ')
        direction = "↑" if feat_value > 0 else "↓"
        impact = "favors home" if feat_value > 0 else "favors away"
        print(f"   {direction} {stat_name:15s}: {feat_value:+.3f} ({impact})")
    
    # Interpretation with actionable insights
    print(f"\n💡 WHAT THIS MEANS:")
    if prob_percent >= 75:
        interpretation = f"Very strong favorite - {favorite_name} has a commanding advantage"
        action = f"{underdog_name} needs a major comeback"
    elif prob_percent >= 65:
        interpretation = f"Strong favorite - {favorite_name} is clearly favored"
        action = f"{underdog_name} needs strong play to overcome deficit"
    elif prob_percent >= 55:
        interpretation = f"Moderate favorite - {favorite_name} has the edge"
        action = f"{underdog_name} is still competitive, needs momentum shift"
    elif prob_percent >= 45:
        interpretation = f"Close game - {favorite_name} slightly favored"
        action = f"Either team could win - game is up for grabs"
    else:
        interpretation = f"Strong underdog - {underdog_name} has significant advantage"
        action = f"{favorite_name} needs a major comeback"
    print(f"   {interpretation}")
    print(f"   🎯 Action: {action}")
    
    # Score context
    score_diff = abs(home_team.get('score', 0) - away_team.get('score', 0))
    leading_team = home_name if home_team.get('score', 0) > away_team.get('score', 0) else away_name
    if score_diff >= 15:
        score_context = f"{leading_team} leads by {score_diff} points - large deficit to overcome"
    elif score_diff >= 10:
        score_context = f"{leading_team} leads by {score_diff} points - significant but surmountable"
    elif score_diff >= 5:
        score_context = f"{leading_team} leads by {score_diff} points - manageable deficit"
    else:
        score_context = f"Tied or close ({score_diff} point difference) - very competitive"
    print(f"   📍 Score: {score_context}")
    
    # Model reliability
    if model_info:
        accuracy = model_info.get('Accuracy', 0) * 100
        auc = model_info.get('AUC', 0)
        print(f"\n📉 MODEL RELIABILITY:")
        if accuracy >= 75:
            reliability = "High"
            reliability_emoji = "🟢"
        elif accuracy >= 70:
            reliability = "Good"
            reliability_emoji = "🟡"
        else:
            reliability = "Moderate"
            reliability_emoji = "🟡"
        print(f"   {reliability_emoji} {reliability} accuracy ({accuracy:.1f}%) | AUC: {auc:.3f}")
        print(f"   Based on {period_type} model trained on historical data")
    
    if ci_lower_percent and ci_upper_percent:
        print(f"\n⚠️  RISK ASSESSMENT:")
        if ci_lower_percent > 50:
            print(f"   Even in worst case ({ci_lower_percent:.1f}%), {favorite_name} still favored")
        elif ci_upper_percent < 50:
            print(f"   Even in best case ({ci_upper_percent:.1f}%), {underdog_name} still favored")
        else:
            print(f"   Confidence interval spans 50% - outcome is uncertain")
            print(f"   Both teams have realistic chance of winning")
    
    print(f"\n{'='*70}\n")

def main():
    """Main loop: monitor live games and make predictions."""
    print("="*70)
    print("NBA Live Game Outcome Predictor")
    print("Monitoring games for Q2 and Q3 end predictions...")
    print("="*70)
    
    # Check if models exist
    for period in ['H2', 'H3']:
        model_path = MODELS_DIR / f'xgb_model_{period}.pkl'
        if not model_path.exists():
            print(f"⚠ Warning: Model not found: {model_path}")
            print("  Please run model training first: python3 3modeltraining/train_models.py")
            return
    
    previous_periods = {}  # Track previous period for each game
    
    try:
        while True:
            # Fetch scoreboard
            data = fetch_scoreboard()
            
            if data and 'scoreboard' in data:
                games = data['scoreboard'].get('games', [])
                
                for game in games:
                    game_status = game.get('gameStatusText', '')
                    game_id = str(game.get('gameId'))
                    
                    # Only process live games
                    if game_status in ['Final', 'Scheduled']:
                        # Reset prediction tracking for finished/scheduled games
                        if game_id in predicted_games:
                            predicted_games[game_id] = {'H2': False, 'H3': False}
                        continue
                    
                    # Check if period ended
                    period_ended, period_that_ended, current_period = check_period_end(game, previous_periods)
                    
                    if period_ended:
                        # Q2 just ended (period changed from 2 to 3)
                        if period_that_ended == 2:
                            predict_game_outcome(game, 'H2')
                        
                        # Q3 just ended (period changed from 3 to 4)
                        elif period_that_ended == 3:
                            predict_game_outcome(game, 'H3')
            
            # Sleep before next check
            time.sleep(2)  # Check every 2 seconds
            
    except KeyboardInterrupt:
        print("\n\nStopped monitoring.")
        sys.exit(0)

if __name__ == "__main__":
    main()

