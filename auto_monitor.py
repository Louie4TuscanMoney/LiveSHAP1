#!/usr/bin/env python3
"""
Auto-Monitoring System for SHAP
Automatically starts monitoring when NBA games are within 2 minutes of starting
and stops monitoring when games conclude.
"""

import requests
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
import json
import sys
import os

# Import monitoring function dynamically (module name starts with digit)
import importlib.util
monitor_path = Path(__file__).parent / '4liveprediction' / 'monitor_live_games.py'
monitor_spec = importlib.util.spec_from_file_location("monitor", monitor_path)
monitor_module = importlib.util.module_from_spec(monitor_spec)
monitor_spec.loader.exec_module(monitor_module)
monitor_main = monitor_module.main
find_live_games = monitor_module.find_live_games

# NBA API endpoint
NBA_API_URL = os.getenv("NBA_API_URL", "https://web-production-8ddddc.up.railway.app")
SCOREBOARD_URL = "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json"

# Track active monitoring threads per game
active_monitors = {}  # {game_id: thread}
monitoring_lock = threading.Lock()

def fetch_nba_scoreboard():
    """Fetch NBA scoreboard data."""
    try:
        response = requests.get(SCOREBOARD_URL, timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching NBA scoreboard: {e}")
        return None

def parse_game_time(game):
    """Parse game start time from NBA API data."""
    # NBA API provides gameDateTimeUTC or gameTimeUTC
    game_time_utc = game.get("gameDateTimeUTC") or game.get("gameTimeUTC")
    if not game_time_utc:
        return None
    
    try:
        # Parse ISO 8601 format: "2025-12-08T20:00:00Z"
        if isinstance(game_time_utc, str):
            # Remove 'Z' and parse
            dt_str = game_time_utc.replace('Z', '+00:00')
            return datetime.fromisoformat(dt_str)
        return None
    except Exception as e:
        print(f"Error parsing game time: {e}")
        return None

def is_game_within_start_window(game, window_minutes=2):
    """Check if game is within window_minutes of starting."""
    game_time = parse_game_time(game)
    if not game_time:
        return False
    
    now = datetime.now(game_time.tzinfo) if game_time.tzinfo else datetime.utcnow()
    time_until_start = (game_time - now).total_seconds() / 60  # minutes
    
    # Game is within window if it starts in 0-2 minutes
    return 0 <= time_until_start <= window_minutes

def is_game_started(game):
    """Check if game has started (not scheduled)."""
    status = game.get("gameStatusText", "")
    period = game.get("period", 0)
    return status != "Scheduled" and period >= 1

def is_game_finished(game):
    """Check if game is finished."""
    status = game.get("gameStatusText", "")
    return status in ["Final", "Final/OT", "Final/2OT", "Final/3OT"]

def start_game_monitoring(game_id, game_info):
    """Start monitoring for a specific game."""
    global monitoring_active
    
    with monitoring_lock:
        if game_id in active_monitors:
            # Already monitoring this game
            return False
        
        print(f"🎯 Game {game_id} ({game_info.get('away_team')} @ {game_info.get('home_team')}) is starting soon - ensuring monitoring is active")
        
        # Track this game
        active_monitors[game_id] = {
            'started_at': datetime.now().isoformat(),
            'game_info': game_info,
            'status': 'pending'
        }
        
        # If monitoring isn't active globally, we need to start it
        # This will be handled by the API server's /api/run endpoint
        # For now, just track the game
        return True

def stop_game_monitoring(game_id):
    """Stop monitoring for a specific game."""
    with monitoring_lock:
        if game_id not in active_monitors:
            return False
        
        print(f"🛑 Stopping monitoring for game {game_id}")
        # Note: We can't easily stop a thread, but we can mark it for cleanup
        # The monitoring will naturally stop when game ends
        del active_monitors[game_id]
        return True

def monitor_single_game(game_id):
    """Monitor a single game."""
    # The existing monitor_main() monitors all live games
    # For per-game monitoring, we'd need to modify monitor_live_games.py
    # For now, we'll use the global monitor which handles all games
    print(f"Game {game_id} started - monitoring active")

def check_and_manage_monitoring():
    """Check NBA games and start/stop monitoring as needed."""
    data = fetch_nba_scoreboard()
    if not data:
        return
    
    games = data.get("scoreboard", {}).get("games", [])
    
    for game in games:
        try:
            game_id = str(game.get("gameId"))
            if not game_id:
                continue
            
            home_team = game.get("homeTeam", {}).get("teamName", "Home")
            away_team = game.get("awayTeam", {}).get("teamName", "Away")
            status = game.get("gameStatusText", "")
            
            game_info = {
                'game_id': game_id,
                'home_team': home_team,
                'away_team': away_team,
                'status': status
            }
            
            # Check if game is within 2 minutes of starting
            if status == "Scheduled" and is_game_within_start_window(game, window_minutes=2):
                if game_id not in active_monitors:
                    start_game_monitoring(game_id, game_info)
                    # Trigger monitoring start (use internal function if available, else API)
                    if hasattr(sys.modules.get('__main__'), 'start_monitoring_func'):
                        # Running in same process - use internal function
                        start_func = sys.modules['__main__'].start_monitoring_func
                        if start_func:
                            start_func()
                            print(f"✅ Monitoring started for upcoming game {game_id}")
                    else:
                        # Running separately - use API
                        api_url = os.getenv("SHAP_API_URL", "http://localhost:5000")
                        try:
                            response = requests.post(f"{api_url}/api/run", timeout=2)
                            if response.status_code == 200:
                                print(f"✅ Monitoring started for upcoming game {game_id}")
                        except Exception as e:
                            print(f"⚠️  Could not start monitoring via API: {e}")
            
            # Check if game has started (ensure monitoring is active)
            elif is_game_started(game):
                if game_id not in active_monitors:
                    start_game_monitoring(game_id, game_info)
                else:
                    # Update status
                    with monitoring_lock:
                        if game_id in active_monitors:
                            active_monitors[game_id]['status'] = 'live'
            
            # Check if game is finished (stop tracking this game)
            elif is_game_finished(game) and game_id in active_monitors:
                stop_game_monitoring(game_id)
                
        except Exception as e:
            print(f"Error processing game: {e}")
            continue

def auto_monitor_loop():
    """Main loop that checks games every 30 seconds."""
    print("="*70)
    print("SHAP AUTO-MONITORING SYSTEM")
    print("Auto-starts monitoring when games are within 2 minutes of starting")
    print("Auto-stops monitoring when games conclude")
    print("="*70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Checking games every 30 seconds...\n")
    
    while True:
        try:
            check_and_manage_monitoring()
            
            # Print status every 5 minutes
            if int(time.time()) % 300 == 0:
                with monitoring_lock:
                    active_count = len(active_monitors)
                    if active_count > 0:
                        print(f"\n📊 Currently monitoring {active_count} game(s):")
                        for game_id, monitor_info in active_monitors.items():
                            game_info = monitor_info.get('game_info', {})
                            print(f"  - {game_info.get('away_team')} @ {game_info.get('home_team')} (started: {monitor_info.get('started_at')})")
                    else:
                        print("\n📊 No games currently being monitored")
            
            time.sleep(30)  # Check every 30 seconds
            
        except KeyboardInterrupt:
            print("\n\nAuto-monitoring stopped.")
            break
        except Exception as e:
            print(f"Error in auto-monitor loop: {e}")
            time.sleep(30)

if __name__ == "__main__":
    auto_monitor_loop()

