#!/usr/bin/env python3
"""
NBA Live Game Monitor
Finds live games and detects when Q2 and Q3 end, then triggers ML predictions.

Checks every 10 seconds until Q2 or Q3 ending is detected.
"""

import requests
import time
import sys
from pathlib import Path
from datetime import datetime

# Import prediction functions from live_predictor
sys.path.append(str(Path(__file__).parent))
from live_predictor import (
    fetch_scoreboard,
    predict_game_outcome,
    MODELS_DIR
)
from prediction_storage import update_actual_outcome

# Scoreboard endpoint
SCOREBOARD_URL = "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json"

# Track game states
game_states = {}  # {game_id: {'period': int, 'status': str, 'h2_predicted': bool, 'h3_predicted': bool}}

def find_live_games():
    """Find all currently live NBA games. Robust error handling for multiple games."""
    try:
        data = fetch_scoreboard()
        if not data or 'scoreboard' not in data:
            return []
        
        games = data['scoreboard'].get('games', [])
        live_games = []
        
        # Process each game individually with error handling
        for game in games:
            try:
                # Validate game data structure
                if not isinstance(game, dict):
                    continue
                
                game_status = game.get('gameStatusText', '')
                period = game.get('period', 0)
                game_id = game.get('gameId')
                
                # Skip if missing critical data
                if not game_id:
                    continue
                
                game_id = str(game_id)
                
                # Consider games as "live" if they're in progress
                if game_status not in ['Final', 'Scheduled'] and period > 0:
                    home_team = game.get('homeTeam', {})
                    away_team = game.get('awayTeam', {})
                    
                    live_games.append({
                        'game': game,
                        'game_id': game_id,
                        'period': period,
                        'status': game_status,
                        'home_team': home_team.get('teamName', 'Home'),
                        'away_team': away_team.get('teamName', 'Away'),
                        'score': f"{away_team.get('score', 0)} - {home_team.get('score', 0)}"
                    })
            except Exception as e:
                # Skip this game if there's an error, continue with others
                print(f"⚠ Skipping game due to error: {e}")
                continue
        
        return live_games
    except Exception as e:
        # If scoreboard fetch fails, return empty list (will retry next cycle)
        print(f"⚠ Error fetching scoreboard: {e}")
        return []

def detect_period_end(game_id, current_period, current_status, game_clock=''):
    """
    Detect if Q2 or Q3 just ended.
    Returns: 'H2' if Q2 just ended, 'H3' if Q3 just ended, None otherwise
    
    Detection logic:
    - Q2 ends when period changes from 2 to 3 OR when period is 2 and clock is 0:00
    - Q3 ends when period changes from 3 to 4 OR when period is 3 and clock is 0:00
    - Also checks game clock to confirm period end (clock at 0:00 or empty)
    - Handles edge cases like period jumps or invalid data
    """
    try:
        # Validate inputs
        if not game_id or not isinstance(current_period, (int, float)):
            return None
        
        # Check if clock indicates period end (handles various formats)
        # IMPORTANT: Only true if clock is ACTUALLY at 0:00 (period ended)
        clock_indicates_end = False
        if game_clock:
            clock_str = str(game_clock).upper()
            # Only match exact 0:00 patterns, not partial matches
            clock_indicates_end = (
                clock_str == '0:00' or 
                clock_str == 'PT0S' or
                clock_str == 'PT00M00.00S' or
                (clock_str.startswith('PT0') and 'M' not in clock_str) or  # PT0S format only
                clock_str == ''
            )
        else:
            clock_indicates_end = True
        
        # Initialize game state if not exists
        if game_id not in game_states:
            game_states[game_id] = {
                'period': int(current_period),
                'status': str(current_status),
                'clock': str(game_clock) if game_clock else '',
                'h2_predicted': False,
                'h3_predicted': False,
                'last_check_time': time.time(),
                'prev_clock': ''  # Start empty to detect if clock already at 0:00
            }
            # Don't return - continue to check if period already ended
            prev_clock = ''
        else:
            prev_state = game_states[game_id]
            prev_clock = prev_state.get('prev_clock', '')
        
        prev_state = game_states[game_id]
        prev_period = prev_state.get('period', 0)
        current_period = int(current_period)
        
        # Handle edge case: period jumped backwards (game reset or data error)
        if current_period < prev_period:
            # Reset prediction flags if period went backwards
            game_states[game_id]['period'] = current_period
            game_states[game_id]['status'] = str(current_status)
            game_states[game_id]['clock'] = str(game_clock) if game_clock else ''
            game_states[game_id]['prev_clock'] = str(game_clock) if game_clock else ''
            return None
        
        # Method 1: Period changed (most reliable)
        if current_period > prev_period and prev_period > 0:
            period_that_ended = int(prev_period)
            
            # Q2 just ended (period changed from 2 to 3)
            if period_that_ended == 2 and not prev_state.get('h2_predicted', False):
                game_states[game_id]['h2_predicted'] = True
                game_states[game_id]['period'] = current_period
                game_states[game_id]['clock'] = str(game_clock) if game_clock else ''
                game_states[game_id]['prev_clock'] = str(game_clock) if game_clock else ''
                game_states[game_id]['last_check_time'] = time.time()
                return 'H2'
            
            # Q3 just ended (period changed from 3 to 4)
            elif period_that_ended == 3 and not prev_state.get('h3_predicted', False):
                game_states[game_id]['h3_predicted'] = True
                game_states[game_id]['period'] = current_period
                game_states[game_id]['clock'] = str(game_clock) if game_clock else ''
                game_states[game_id]['prev_clock'] = str(game_clock) if game_clock else ''
                game_states[game_id]['last_check_time'] = time.time()
                return 'H3'
        
        # Method 2: Clock indicates end but period hasn't updated yet
        # Simple logic: If period is 2/3 and clock is 0:00, period has ended
        # Check if clock just changed to 0:00 (wasn't 0:00 before, now is)
        clock_just_ended = (
            clock_indicates_end and 
            prev_clock and 
            prev_clock.upper() not in ['0:00', 'PT0S', 'PT00M00.00S', ''] and
            '00M00' not in prev_clock.upper()
        )
        
        # Q2 end detection: period is 2, clock is 0:00, haven't predicted yet
        if current_period == 2 and clock_indicates_end and not prev_state.get('h2_predicted', False):
            # Trigger if:
            # 1. Clock just changed to 0:00, OR
            # 2. Clock is 0:00 and we haven't seen a non-zero clock yet (script started mid-game)
            if clock_just_ended or (not prev_clock or prev_clock == ''):
                game_states[game_id]['h2_predicted'] = True
                game_states[game_id]['period'] = current_period
                game_states[game_id]['clock'] = str(game_clock) if game_clock else ''
                game_states[game_id]['prev_clock'] = str(game_clock) if game_clock else ''
                game_states[game_id]['last_check_time'] = time.time()
                print(f"  [DEBUG] Q2 detected: period={current_period}, clock={game_clock}, prev_clock={prev_clock}, clock_just_ended={clock_just_ended}")
                return 'H2'
        
        # Q3 end detection: period is 3, clock is 0:00, haven't predicted yet
        # CRITICAL: Only trigger if clock is ACTUALLY at 0:00 (not counting down)
        if current_period == 3 and clock_indicates_end and not prev_state.get('h3_predicted', False):
            # Only trigger if:
            # 1. Clock is actually at 0:00 (PT00M00.00S or similar)
            # 2. Previous clock was NOT 0:00 (period was in progress)
            # This prevents false triggers when Q3 starts (clock resets to 12:00)
            prev_was_not_zero = (
                prev_clock and 
                prev_clock.upper() not in ['0:00', 'PT0S', 'PT00M00.00S', ''] and
                not (prev_clock.upper().startswith('PT0') and 'M' not in prev_clock.upper())
            )
            # Only trigger if clock actually hit 0:00 (period ended)
            if prev_was_not_zero and clock_indicates_end:
                game_states[game_id]['h3_predicted'] = True
                game_states[game_id]['period'] = current_period
                game_states[game_id]['clock'] = str(game_clock) if game_clock else ''
                game_states[game_id]['prev_clock'] = str(game_clock) if game_clock else ''
                game_states[game_id]['last_check_time'] = time.time()
                print(f"  [DEBUG] Q3 detected: period={current_period}, clock={game_clock}, prev_clock={prev_clock}")
                return 'H3'
        
        # Update stored period and clock
        game_states[game_id]['period'] = current_period
        game_states[game_id]['status'] = str(current_status)
        game_states[game_id]['clock'] = str(game_clock) if game_clock else ''
        game_states[game_id]['prev_clock'] = str(game_clock) if game_clock else ''
        game_states[game_id]['last_check_time'] = time.time()
        
        return None
    except Exception as e:
        # If detection fails, log and return None (don't crash)
        print(f"⚠ Error detecting period end for game {game_id}: {e}")
        return None

def check_game_clock_for_period_end(game):
    """
    Alternative method: Check game clock to detect period end.
    If clock shows 0:00 and period is 2 or 3, period just ended.
    """
    game_clock = game.get('gameClock', '')
    period = game.get('period', 0)
    game_status = game.get('gameStatusText', '')
    
    # If game is live and clock shows 0:00 or empty, period might have ended
    if game_status not in ['Final', 'Scheduled']:
        # Check if clock is at 0:00 or period just changed
        if game_clock in ['', '0:00', 'PT0S'] or not game_clock:
            # This might indicate period end, but we rely on period change detection
            pass
    
    return None

def display_live_games(live_games):
    """Display currently live games."""
    if not live_games:
        return
    
    print(f"\n{'='*70}")
    print(f"LIVE GAMES ({len(live_games)} game{'s' if len(live_games) != 1 else ''})")
    print(f"{'='*70}")
    
    for game_info in live_games:
        game = game_info['game']
        period = game.get('period', 0)
        game_clock = game.get('gameClock', '')
        status = game.get('gameStatusText', '')
        
        # Format clock
        clock_display = game_clock if game_clock else status
        
        print(f"  {game_info['away_team']} @ {game_info['home_team']}")
        print(f"    Score: {game_info['score']} | Period: Q{period} | Clock: {clock_display}")
        
        # Show prediction status
        game_id = game_info['game_id']
        if game_id in game_states:
            state = game_states[game_id]
            predictions = []
            if state.get('h2_predicted'):
                predictions.append('H2 ✓')
            if state.get('h3_predicted'):
                predictions.append('H3 ✓')
            if predictions:
                print(f"    Predictions: {', '.join(predictions)}")
    
    print(f"{'='*70}\n")

def main():
    """Main monitoring loop."""
    print("="*70)
    print("NBA LIVE GAME MONITOR")
    print("Finding live games and detecting Q2/Q3 end for ML predictions")
    print("="*70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Checking every 10 seconds until Q2 or Q3 ending detected...")
    print("Press Ctrl+C to stop\n")
    
    # Check if models exist
    h2_model = MODELS_DIR / 'xgb_model_H2.pkl'
    h3_model = MODELS_DIR / 'xgb_model_H3.pkl'
    
    if not h2_model.exists():
        print(f"⚠ ERROR: Model not found: {h2_model}")
        print("  Please run: python3 3modeltraining/train_models.py")
        sys.exit(1)
    
    if not h3_model.exists():
        print(f"⚠ ERROR: Model not found: {h3_model}")
        print("  Please run: python3 3modeltraining/train_models.py")
        sys.exit(1)
    
    print("✓ Models found. Starting monitoring...\n")
    
    last_display_time = 0
    display_interval = 30  # Display live games every 30 seconds
    
    consecutive_errors = 0
    max_consecutive_errors = 5
    
    try:
        while True:
            try:
                # Find live games (with error handling)
                live_games = find_live_games()
                
                # Reset error counter on success
                consecutive_errors = 0
                
                # Display live games periodically
                current_time = time.time()
                if current_time - last_display_time >= display_interval:
                    try:
                        display_live_games(live_games)
                    except Exception as e:
                        print(f"⚠ Error displaying games: {e}")
                    last_display_time = current_time
                
                # Check each live game for period end (process independently)
                for game_info in live_games:
                    try:
                        game = game_info['game']
                        game_id = game_info['game_id']
                        current_period = game_info['period']
                        current_status = game_info['status']
                        game_clock = game.get('gameClock', '')
                        
                        # Validate period is reasonable (1-4 for normal game)
                        if not isinstance(current_period, (int, float)) or current_period < 1 or current_period > 10:
                            print(f"  [DEBUG] Skipping game {game_id}: invalid period {current_period}")
                            continue
                        
                        # Debug: Print detection attempt (only first few times to avoid spam)
                        if game_id not in game_states or len([k for k in game_states.keys() if k == game_id]) < 3:
                            print(f"  [DEBUG] Checking game {game_id}: period={current_period}, clock={game_clock}, status={current_status}")
                        
                        # Detect if Q2 or Q3 just ended
                        period_ended = detect_period_end(game_id, current_period, current_status, game_clock)
                        
                        if period_ended:
                            print(f"  [DEBUG] ✓ Period end detected: {period_ended} for game {game_id}")
                        
                        if period_ended:
                            print(f"\n{'='*70}")
                            print(f"🎯 PERIOD END DETECTED: {period_ended}")
                            print(f"   Game: {game_info['away_team']} @ {game_info['home_team']}")
                            print(f"   Period: Q{current_period - 1} just ended → Q{current_period}")
                            print(f"   Score: {game_info['score']}")
                            print(f"{'='*70}\n")
                            
                            # Trigger ML prediction (with error handling)
                            try:
                                predict_game_outcome(game, period_ended)
                            except Exception as e:
                                print(f"⚠ Error making prediction for {game_info['away_team']} @ {game_info['home_team']}: {e}\n")
                                # Continue monitoring other games even if this prediction fails
                                import traceback
                                traceback.print_exc()
                    except Exception as e:
                        # Skip this game if there's an error, continue with others
                        print(f"⚠ Error processing game {game_info.get('game_id', 'unknown')}: {e}")
                        continue
                
                # Clean up finished games from tracking and update outcomes
                finished_game_ids = []
                for game_info in live_games:
                    game = game_info['game']
                    game_id = game_info['game_id']
                    game_status = game_info['status']
                    
                    # If game is finished, update actual outcome
                    if game_status == 'Final':
                        try:
                            home_team = game.get('homeTeam', {})
                            away_team = game.get('awayTeam', {})
                            home_score = home_team.get('score', 0)
                            away_score = away_team.get('score', 0)
                            
                            # Determine outcome: 1 = home win, 0 = home loss
                            actual_outcome = 1 if home_score > away_score else 0
                            update_actual_outcome(game_id, actual_outcome)
                        except Exception as e:
                            print(f"⚠ Error updating outcome for game {game_id}: {e}")
                
                # Remove finished games from tracking
                for game_id, state in game_states.items():
                    # Check if game is still live
                    is_still_live = any(
                        g['game_id'] == game_id 
                        for g in live_games
                    )
                    if not is_still_live:
                        finished_game_ids.append(game_id)
                
                for game_id in finished_game_ids:
                    del game_states[game_id]
                
                # Sleep before next check
                time.sleep(10)  # Check every 10 seconds
                    
            except Exception as e:
                consecutive_errors += 1
                print(f"⚠ Error in main loop (attempt {consecutive_errors}/{max_consecutive_errors}): {e}")
                
                # If too many consecutive errors, wait longer before retrying
                if consecutive_errors >= max_consecutive_errors:
                    print(f"⚠ Too many errors. Waiting 30 seconds before retry...")
                    time.sleep(30)
                    consecutive_errors = 0
                else:
                    time.sleep(5)  # Short wait before retry
                continue
            
    except KeyboardInterrupt:
        print("\n\n" + "="*70)
        print("Monitoring stopped.")
        print(f"Stopped at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
        sys.exit(0)

if __name__ == "__main__":
    main()

