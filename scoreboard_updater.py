#!/usr/bin/env python3
"""NBA Scoreboard Updater - Minimal output, optimized for speed."""

import requests
import time
import sys
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

SCOREBOARD_URL = "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json"

# Cache for possession data (updated less frequently)
possession_cache = {}
possession_lock = threading.Lock()
possession_update_in_progress = False
possession_update_lock = threading.Lock()

def fetch_scoreboard():
    """Fetch scoreboard data from NBA API."""
    try:
        response = requests.get(SCOREBOARD_URL, timeout=2)
        response.raise_for_status()
        return response.json()
    except:
        return None

def get_possession_single(game_id):
    """Fetch playbyplay data to get current possession for a single game."""
    try:
        url = f"https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_{game_id}.json"
        response = requests.get(url, timeout=1.5)
        response.raise_for_status()
        data = response.json()
        actions = data.get("game", {}).get("actions", [])
        if actions:
            last_action = actions[-1]
            return game_id, last_action.get("possession")
    except:
        pass
    return game_id, None

def update_possession_cache(live_game_ids):
    """Fetch possession for all live games in parallel."""
    global possession_update_in_progress
    if not live_game_ids:
        return
    
    with possession_update_lock:
        if possession_update_in_progress:
            return  # Skip if update already in progress
        possession_update_in_progress = True
    
    try:
        with ThreadPoolExecutor(max_workers=len(live_game_ids)) as executor:
            futures = {executor.submit(get_possession_single, game_id): game_id for game_id in live_game_ids}
            for future in as_completed(futures):
                game_id, possession = future.result()
                with possession_lock:
                    if possession:
                        possession_cache[game_id] = possession
                    elif game_id in possession_cache:
                        # Keep old possession if fetch fails
                        pass
    finally:
        with possession_update_lock:
            possession_update_in_progress = False

def format_clock(game):
    """Format game clock to show time remaining in quarter."""
    game_clock = game.get("gameClock", "")
    game_status = game.get("gameStatusText", "")
    period = game.get("period", 0)
    
    # If game is live and has a clock
    if game_clock and game_status not in ["Final", "Scheduled"]:
        # Handle ISO 8601 duration format like "PT01M50.00S"
        if game_clock.startswith("PT"):
            try:
                # Parse PT01M50.00S format
                match = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:([\d.]+)S)?', game_clock)
                if match:
                    hours = int(match.group(1) or 0)
                    minutes = int(match.group(2) or 0)
                    seconds = float(match.group(3) or 0)
                    total_seconds = hours * 3600 + minutes * 60 + int(seconds)
                    mins = total_seconds // 60
                    secs = total_seconds % 60
                    return f"Q{period} {int(mins)}:{int(secs):02d}"
            except:
                pass
        
        # Handle MM:SS format
        if ":" in game_clock:
            return f"Q{period} {game_clock}"
        
        # Handle other formats
        return f"Q{period} {game_clock}"
    
    # For scheduled games, show start time
    if game_status == "Scheduled" or not game_clock:
        game_et = game.get("gameEt", "")
        if game_et:
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(game_et.replace('Z', '+00:00'))
                return dt.strftime("%I:%M %p ET")
            except:
                pass
        return game_status if game_status else ""
    
    # For final games
    if game_status == "Final":
        return f"Final Q{period}"
    
    return game_status

def display_scoreboard(data):
    """Display scores with possession indicator and game clock."""
    if not data or "scoreboard" not in data:
        return
    
    games = data["scoreboard"].get("games", [])
    if not games:
        return
    
    output = []
    for game in games:
        home = game.get("homeTeam", {})
        away = game.get("awayTeam", {})
        away_initials = away.get("teamTricode", "")
        home_initials = home.get("teamTricode", "")
        away_score = away.get("score", 0)
        home_score = home.get("score", 0)
        home_team_id = home.get("teamId")
        away_team_id = away.get("teamId")
        
        # Format clock
        clock_display = format_clock(game)
        
        # Get possession from cache (updated separately)
        away_star = ""
        home_star = ""
        game_status = game.get("gameStatusText", "")
        if game_status not in ["Final", "Scheduled"] and game.get("gameClock"):
            game_id = game.get("gameId")
            with possession_lock:
                possession_team_id = possession_cache.get(game_id)
            if possession_team_id:
                if possession_team_id == home_team_id:
                    home_star = "*"
                elif possession_team_id == away_team_id:
                    away_star = "*"
        
        output.append(f"{away_initials}{away_star} {away_score} {home_initials}{home_star} {home_score} {clock_display}")
    
    print("\033[H\033[J" + "\n".join(output), end="", flush=True)

def main():
    try:
        possession_update_counter = 0
        while True:
            # Always fetch scoreboard (fast, has scores and clock)
            data = fetch_scoreboard()
            if data:
                # Get list of live game IDs for possession updates
                live_game_ids = []
                games = data.get("scoreboard", {}).get("games", [])
                for game in games:
                    game_status = game.get("gameStatusText", "")
                    if game_status not in ["Final", "Scheduled"] and game.get("gameClock"):
                        live_game_ids.append(game.get("gameId"))
                
                # Update possession cache every 0.15 seconds (hyperoptimized - fastest possible)
                # This runs in background thread so it doesn't block scoreboard display
                if possession_update_counter % 1 == 0:  # Every iteration (0.15s)
                    threading.Thread(target=update_possession_cache, args=(live_game_ids,), daemon=True).start()
                
                display_scoreboard(data)
                possession_update_counter += 1
            
            time.sleep(0.15)  # Hyperoptimized: 0.15s updates (fastest possible)
    except KeyboardInterrupt:
        print("\n")
        sys.exit(0)

if __name__ == "__main__":
    main()

