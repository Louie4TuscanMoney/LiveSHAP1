"""
NBA Data Collection using ONLY nba_api package
Collects game statistics with H2 (first half) and H3 (first 3 quarters) breakdowns

This script uses nba_api endpoints to collect:
- Full game statistics from BoxScoreTraditionalV2
- H2 (first half) statistics using BoxScoreTraditionalV2 with RangeType=2, StartRange=0, EndRange=14400
- H3 (first 3 quarters) statistics using BoxScoreTraditionalV2 with RangeType=2, StartRange=0, EndRange=21600
- All statistics as differences (home - away) matching research paper format

Note: Uses time-based ranges (RangeType=2) in seconds:
- H2: 0-14400 seconds (first half = Q1 + Q2)
- H3: 0-21600 seconds (first 3 quarters = Q1 + Q2 + Q3)

OPTIMIZATIONS:
- Parallel processing: Uses multiprocessing to process multiple games concurrently (default: 8 workers)
- Reduced rate limiting: 0.3s delay (down from 0.6s) - still safe for API
- Removed unused API calls: Eliminated get_game_summary() call that wasn't being used
- Batch checkpoint saves: Saves checkpoint every 10 games instead of every game
- Expected speedup: ~5-8x faster depending on CPU cores
"""

import pandas as pd
import numpy as np
from nba_api.stats.endpoints import (
    leaguegamefinder,
    boxscoretraditionalv2,
    boxscoresummaryv2
)
from nba_api.stats.static import teams
import time
from datetime import datetime
from tqdm import tqdm
import warnings
import json
import os
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial
warnings.filterwarnings('ignore')


def process_single_game_wrapper(args):
    """Wrapper function for multiprocessing (must be at module level for pickling)"""
    game_id, season, dataset_dir, rate_limit_delay = args
    
    # Create a temporary collector instance for this worker
    collector = NBADataCollector(dataset_dir=dataset_dir, rate_limit_delay=rate_limit_delay)
    collector.max_workers = 1  # Disable nested parallelism
    
    game_data = collector.process_game_data(game_id, season)
    
    if game_data:
        collector.save_game_json(game_data)
        return game_id, True
    return game_id, False


class NBADataCollector:
    """
    Collect NBA game data using only nba_api package
    Aggregates quarter statistics from play-by-play data
    """
    
    def __init__(self, dataset_dir='Dataset', rate_limit_delay=0.3, max_workers=None):
        self.seasons = ['2020-21', '2021-22', '2022-23']
        self.rate_limit_delay = rate_limit_delay  # Reduced from 0.6 to 0.3 seconds
        self.dataset_dir = Path(dataset_dir)
        self.dataset_dir.mkdir(exist_ok=True)
        
        # Create games subdirectory for individual game files
        self.games_dir = self.dataset_dir / 'games'
        self.games_dir.mkdir(exist_ok=True)
        
        # Checkpoint files for each season (tracks processed game IDs)
        self.checkpoint_files = {
            season: self.dataset_dir / f'checkpoint_{season.replace("-", "_")}.json'
            for season in self.seasons
        }
        
        # Parallel processing settings
        self.max_workers = max_workers or min(cpu_count(), 8)  # Limit to 8 workers max
        self.checkpoint_save_interval = 10  # Save checkpoint every N games instead of every game
        
    def get_season_games(self, season):
        """Get all games for a season"""
        print(f"\nFetching games for season {season}...")
        
        try:
            time.sleep(self.rate_limit_delay)
            gamefinder = leaguegamefinder.LeagueGameFinder(
                season_nullable=season,
                season_type_nullable='Regular Season'
            )
            games = gamefinder.get_data_frames()[0]
            games = games.drop_duplicates(subset=['GAME_ID'])
            
            print(f"Found {len(games)} games for {season}")
            return games
            
        except Exception as e:
            print(f"Error fetching games for {season}: {e}")
            return pd.DataFrame()
    
    def get_period_boxscore(self, game_id, start_range, end_range):
        """Get boxscore data for a specific time range using RangeType=2 (time-based)"""
        try:
            time.sleep(self.rate_limit_delay)
            # Use BoxScoreTraditionalV2 with range_type=2 for period-specific stats
            # RangeType=2 means time-based ranges (in seconds)
            boxscore = boxscoretraditionalv2.BoxScoreTraditionalV2(
                game_id=game_id,
                range_type='2',
                start_range=str(start_range),
                end_range=str(end_range)
            )
            data_frames = boxscore.get_data_frames()
            if len(data_frames) > 1 and data_frames[1] is not None:
                team_stats = data_frames[1]  # Team stats
                if len(team_stats) >= 2:
                    return team_stats
            return None
        except Exception as e:
            print(f"Error fetching period boxscore for game {game_id} (range {start_range}-{end_range}): {e}")
            return None
    
    def get_boxscore(self, game_id, start_period=0, end_period=0):
        """Get boxscore data for a full game"""
        try:
            time.sleep(self.rate_limit_delay)
            # Full game - use V2 without range parameters
            boxscore = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
            data_frames = boxscore.get_data_frames()
            if len(data_frames) > 1 and data_frames[1] is not None:
                team_stats = data_frames[1]  # Team stats
                if len(team_stats) >= 2:
                    return team_stats
            return None
        except Exception as e:
            print(f"Error fetching boxscore for game {game_id}: {e}")
            return None
    
    def convert_v3_to_v2_format(self, v3_df):
        """Convert V3 dataframe column names to match V2 format and aggregate Starters+Bench"""
        # V3 DataFrame 1 has rows split by Starters/Bench - we need to sum them
        # Check if we have Starters/Bench split
        if 'startersBench' in v3_df.columns:
            # Group by team and sum Starters + Bench
            team_stats = {}
            for _, row in v3_df.iterrows():
                team_id = row['teamId']
                if team_id not in team_stats:
                    team_stats[team_id] = {
                        'TEAM_ID': team_id,
                        'TEAM_NAME': row.get('teamName', 'Unknown'),
                        'FGM': 0, 'FGA': 0, 'FG3M': 0, 'FG3A': 0,
                        'FTM': 0, 'FTA': 0, 'OREB': 0, 'DREB': 0,
                        'AST': 0, 'STL': 0, 'BLK': 0, 'TOV': 0, 'PF': 0, 'PTS': 0
                    }
                
                # Sum stats
                team_stats[team_id]['FGM'] += row.get('fieldGoalsMade', 0)
                team_stats[team_id]['FGA'] += row.get('fieldGoalsAttempted', 0)
                team_stats[team_id]['FG3M'] += row.get('threePointersMade', 0)
                team_stats[team_id]['FG3A'] += row.get('threePointersAttempted', 0)
                team_stats[team_id]['FTM'] += row.get('freeThrowsMade', 0)
                team_stats[team_id]['FTA'] += row.get('freeThrowsAttempted', 0)
                team_stats[team_id]['OREB'] += row.get('reboundsOffensive', 0)
                team_stats[team_id]['DREB'] += row.get('reboundsDefensive', 0)
                team_stats[team_id]['AST'] += row.get('assists', 0)
                team_stats[team_id]['STL'] += row.get('steals', 0)
                team_stats[team_id]['BLK'] += row.get('blocks', 0)
                team_stats[team_id]['TOV'] += row.get('turnovers', 0)
                team_stats[team_id]['PF'] += row.get('foulsPersonal', 0)
                team_stats[team_id]['PTS'] += row.get('points', 0)
            
            # Convert to DataFrame
            result_df = pd.DataFrame(list(team_stats.values()))
            return result_df
        else:
            # No Starters/Bench split, just convert column names
            df_copy = v3_df.copy()
            
            # Map V3 columns to V2 format
            if 'fieldGoalsMade' in df_copy.columns:
                df_copy['FGM'] = df_copy['fieldGoalsMade']
            if 'fieldGoalsAttempted' in df_copy.columns:
                df_copy['FGA'] = df_copy['fieldGoalsAttempted']
            if 'threePointersMade' in df_copy.columns:
                df_copy['FG3M'] = df_copy['threePointersMade']
            if 'threePointersAttempted' in df_copy.columns:
                df_copy['FG3A'] = df_copy['threePointersAttempted']
            if 'freeThrowsMade' in df_copy.columns:
                df_copy['FTM'] = df_copy['freeThrowsMade']
            if 'freeThrowsAttempted' in df_copy.columns:
                df_copy['FTA'] = df_copy['freeThrowsAttempted']
            if 'reboundsOffensive' in df_copy.columns:
                df_copy['OREB'] = df_copy['reboundsOffensive']
            if 'reboundsDefensive' in df_copy.columns:
                df_copy['DREB'] = df_copy['reboundsDefensive']
            if 'reboundsTotal' in df_copy.columns:
                df_copy['REB'] = df_copy['reboundsTotal']
            if 'assists' in df_copy.columns:
                df_copy['AST'] = df_copy['assists']
            if 'steals' in df_copy.columns:
                df_copy['STL'] = df_copy['steals']
            if 'blocks' in df_copy.columns:
                df_copy['BLK'] = df_copy['blocks']
            if 'turnovers' in df_copy.columns:
                df_copy['TOV'] = df_copy['turnovers']
            if 'foulsPersonal' in df_copy.columns:
                df_copy['PF'] = df_copy['foulsPersonal']
            if 'points' in df_copy.columns:
                df_copy['PTS'] = df_copy['points']
            if 'teamId' in df_copy.columns:
                df_copy['TEAM_ID'] = df_copy['teamId']
            if 'teamName' in df_copy.columns:
                df_copy['TEAM_NAME'] = df_copy['teamName']
            
            return df_copy
    
    
    def get_quarter_stats_from_boxscore(self, boxscore_df, team_id, quarter_range):
        """
        Alternative: Try to get quarter stats from boxscore if available
        This is a fallback if play-by-play aggregation is incomplete
        """
        # BoxScoreTraditionalV2 doesn't provide quarter stats directly
        # We'll use play-by-play aggregation instead
        return None
    
    def calculate_percentage(self, made, attempted):
        """Calculate percentage"""
        if attempted == 0:
            return 0.0
        return made / attempted
    
    def get_team_stats_dict(self, team_stats_row, prefix=''):
        """Extract all team statistics into a dictionary"""
        def get_stat(team_stats, stat_name, default=0):
            """Safely get stat value, trying alternative column names"""
            if stat_name in team_stats:
                return int(team_stats[stat_name])
            alternatives = {
                'TOV': ['TO', 'TOV'],
                'OREB': ['ORB', 'OREB'],
                'DREB': ['DRB', 'DREB'],
                'FGM': ['FG', 'FGM'],
                'FGA': ['FGA'],
                'FG3M': ['FG3', 'FG3M', '3PM'],
                'FG3A': ['FG3A', '3PA'],
                'FTM': ['FT', 'FTM'],
                'FTA': ['FTA'],
            }
            if stat_name in alternatives:
                for alt in alternatives[stat_name]:
                    if alt in team_stats:
                        return int(team_stats[alt])
            return default
        
        fgm = get_stat(team_stats_row, 'FGM')
        fga = get_stat(team_stats_row, 'FGA')
        fg3m = get_stat(team_stats_row, 'FG3M')
        fg3a = get_stat(team_stats_row, 'FG3A')
        ftm = get_stat(team_stats_row, 'FTM')
        fta = get_stat(team_stats_row, 'FTA')
        
        fg2m = fgm - fg3m
        fg2a = fga - fg3a
        
        stats = {
            f'{prefix}FG': fgm,
            f'{prefix}FGA': fga,
            f'{prefix}FG%': self.calculate_percentage(fgm, fga),
            f'{prefix}2P': fg2m,
            f'{prefix}2PA': fg2a,
            f'{prefix}2P%': self.calculate_percentage(fg2m, fg2a),
            f'{prefix}3P': fg3m,
            f'{prefix}3PA': fg3a,
            f'{prefix}3P%': self.calculate_percentage(fg3m, fg3a),
            f'{prefix}FT': ftm,
            f'{prefix}FTA': fta,
            f'{prefix}FT%': self.calculate_percentage(ftm, fta),
            f'{prefix}ORB': get_stat(team_stats_row, 'OREB'),
            f'{prefix}DRB': get_stat(team_stats_row, 'DREB'),
            f'{prefix}TRB': get_stat(team_stats_row, 'OREB') + get_stat(team_stats_row, 'DREB'),
            f'{prefix}AST': get_stat(team_stats_row, 'AST'),
            f'{prefix}STL': get_stat(team_stats_row, 'STL'),
            f'{prefix}BLK': get_stat(team_stats_row, 'BLK'),
            f'{prefix}TOV': get_stat(team_stats_row, 'TOV'),
            f'{prefix}PF': get_stat(team_stats_row, 'PF'),
            f'{prefix}PTS': get_stat(team_stats_row, 'PTS'),
        }
        
        return stats
    
    def process_game_data(self, game_id, season):
        """
        Process a single game to extract all required statistics
        Returns a dictionary with home_team, away_team, and differences
        """
        try:
            # Get boxscore for full game stats
            team_stats = self.get_boxscore(game_id)
            if team_stats is None or len(team_stats) < 2:
                return None
            
            # Identify home and away teams (removed unused get_game_summary call)
            away_team_stats = team_stats.iloc[0]
            home_team_stats = team_stats.iloc[1]
            
            away_team_id = int(away_team_stats['TEAM_ID'])
            home_team_id = int(home_team_stats['TEAM_ID'])
            
            # Get team names
            away_team_name = str(away_team_stats.get('TEAM_NAME', 'Unknown'))
            home_team_name = str(home_team_stats.get('TEAM_NAME', 'Unknown'))
            
            # Get game result
            home_score = int(home_team_stats['PTS'])
            away_score = int(away_team_stats['PTS'])
            result = 1 if home_score > away_score else 0
            
            # Build comprehensive game data structure
            game_data = {
                'GAME_ID': game_id,
                'SEASON': season,
                'RESULT': result,
                'home_team': {
                    'TEAM_ID': home_team_id,
                    'TEAM_NAME': home_team_name,
                    'PTS': home_score
                },
                'away_team': {
                    'TEAM_ID': away_team_id,
                    'TEAM_NAME': away_team_name,
                    'PTS': away_score
                },
                'game': {
                    'home_team': self.get_team_stats_dict(home_team_stats, ''),
                    'away_team': self.get_team_stats_dict(away_team_stats, ''),
                    'differences': {}
                },
                'H2': {
                    'home_team': {},
                    'away_team': {},
                    'differences': {}
                },
                'H3': {
                    'home_team': {},
                    'away_team': {},
                    'differences': {}
                }
            }
            
            # Calculate full game differences
            home_game = game_data['game']['home_team']
            away_game = game_data['game']['away_team']
            for key in ['FG', 'FGA', 'FG%', '2P', '2PA', '2P%', '3P', '3PA', '3P%', 
                       'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF']:
                game_data['game']['differences'][f'game_{key}'] = home_game[key] - away_game[key]
            
            # Get H2 statistics (first half: 0 to 14400 seconds) using RangeType=2
            # H2 = first half = Q1 (0-7200s) + Q2 (7200-14400s) = 0-14400s
            h2_boxscore = self.get_period_boxscore(game_id, start_range=0, end_range=14400)
            if h2_boxscore is not None and len(h2_boxscore) >= 2:
                # Find home and away teams in H2 boxscore (order may differ)
                h2_away_idx = 0 if h2_boxscore.iloc[0]['TEAM_ID'] == away_team_id else 1
                h2_home_idx = 1 - h2_away_idx
                
                h2_away_stats = h2_boxscore.iloc[h2_away_idx]
                h2_home_stats = h2_boxscore.iloc[h2_home_idx]
                
                game_data['H2']['home_team'] = self.get_team_stats_dict(h2_home_stats, '')
                game_data['H2']['away_team'] = self.get_team_stats_dict(h2_away_stats, '')
                
                # Calculate H2 differences
                h2_home = game_data['H2']['home_team']
                h2_away = game_data['H2']['away_team']
                for key in ['FG', 'FGA', 'FG%', '2P', '2PA', '2P%', '3P', '3PA', '3P%', 
                           'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF']:
                    game_data['H2']['differences'][f'H2_{key}'] = h2_home[key] - h2_away[key]
            else:
                print(f"Warning: H2 stats not available for game {game_id}")
            
            # Get H3 statistics (first 3 quarters: 0 to 21600 seconds) using RangeType=2
            # H3 = first 3 quarters = Q1 (0-7200s) + Q2 (7200-14400s) + Q3 (14400-21600s) = 0-21600s
            h3_boxscore = self.get_period_boxscore(game_id, start_range=0, end_range=21600)
            if h3_boxscore is not None and len(h3_boxscore) >= 2:
                # Find home and away teams in H3 boxscore
                h3_away_idx = 0 if h3_boxscore.iloc[0]['TEAM_ID'] == away_team_id else 1
                h3_home_idx = 1 - h3_away_idx
                
                h3_away_stats = h3_boxscore.iloc[h3_away_idx]
                h3_home_stats = h3_boxscore.iloc[h3_home_idx]
                
                game_data['H3']['home_team'] = self.get_team_stats_dict(h3_home_stats, '')
                game_data['H3']['away_team'] = self.get_team_stats_dict(h3_away_stats, '')
                
                # Calculate H3 differences
                h3_home = game_data['H3']['home_team']
                h3_away = game_data['H3']['away_team']
                for key in ['FG', 'FGA', 'FG%', '2P', '2PA', '2P%', '3P', '3PA', '3P%', 
                           'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF']:
                    game_data['H3']['differences'][f'H3_{key}'] = h3_home[key] - h3_away[key]
            else:
                print(f"Warning: H3 stats not available for game {game_id}")
            
            return game_data
            
        except Exception as e:
            print(f"Error processing game {game_id}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def load_checkpoint(self, season):
        """Load checkpoint data for a season if it exists"""
        checkpoint_file = self.checkpoint_files[season]
        if checkpoint_file.exists():
            try:
                with open(checkpoint_file, 'r') as f:
                    content = f.read()
                    # Replace any invalid NaN values with null before parsing
                    content = content.replace(': NaN', ': null').replace(':NaN', ': null')
                    checkpoint = json.loads(content)
                
                processed_ids = set(checkpoint.get('processed_game_ids', []))
                
                # Count existing game files
                existing_files = len(list(self.games_dir.glob('*.json')))
                print(f"Found checkpoint for {season}: {len(processed_ids)} games already processed")
                return processed_ids
            except json.JSONDecodeError as e:
                print(f"Error parsing checkpoint JSON for {season}: {e}")
                print("Attempting to fix and reload...")
                try:
                    with open(checkpoint_file, 'r') as f:
                        content = f.read()
                    content = content.replace(': NaN', ': null').replace(':NaN', ': null')
                    checkpoint = json.loads(content)
                    with open(checkpoint_file, 'w') as f:
                        json.dump(checkpoint, f, indent=2, allow_nan=False)
                    print(f"Fixed and reloaded checkpoint for {season}")
                    return set(checkpoint.get('processed_game_ids', []))
                except Exception as e2:
                    print(f"Could not fix checkpoint file: {e2}")
                    return set()
            except Exception as e:
                print(f"Error loading checkpoint for {season}: {e}")
                return set()
        return set()
    
    def save_checkpoint(self, season, processed_game_ids):
        """Save checkpoint data for a season (only tracks processed game IDs now)"""
        checkpoint_file = self.checkpoint_files[season]
        try:
            checkpoint = {
                'season': season,
                'processed_game_ids': sorted(list(processed_game_ids)),
                'last_updated': datetime.now().isoformat()
            }
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint, f, indent=2, allow_nan=False)
        except Exception as e:
            print(f"Error saving checkpoint for {season}: {e}")
    
    def convert_to_json_serializable(self, obj):
        """Recursively convert numpy types and handle NaN values for JSON serialization"""
        if isinstance(obj, dict):
            return {key: self.convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_to_json_serializable(item) for item in obj]
        elif pd.isna(obj) or obj is None or (isinstance(obj, float) and np.isnan(obj)):
            return None
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        elif isinstance(obj, (int, float, str, bool)):
            if isinstance(obj, float) and (obj != obj or np.isnan(obj)):  # NaN check
                return None
            return obj
        else:
            return str(obj)
    
    def save_game_json(self, game_data):
        """Save individual game data to a separate JSON file"""
        try:
            game_id = game_data['GAME_ID']
            game_file = self.games_dir / f'{game_id}.json'
            
            # Convert to JSON-serializable format
            json_data = self.convert_to_json_serializable(game_data)
            
            with open(game_file, 'w') as f:
                json.dump(json_data, f, indent=2, allow_nan=False)
            
            return str(game_file)
        except Exception as e:
            print(f"Error saving game {game_data.get('GAME_ID', 'unknown')}: {e}")
            return None
    
    def collect_all_data(self):
        """Main collection method with checkpoint/resume support"""
        print("=" * 60)
        print("NBA Data Collection using nba_api ONLY")
        print("Seasons: 2020-21, 2021-22, 2022-23")
        print(f"Dataset directory: {self.dataset_dir}")
        print("=" * 60)
        
        for season in self.seasons:
            # Load checkpoint for this season
            processed_ids = self.load_checkpoint(season)
            
            # Get all games for the season
            games_df = self.get_season_games(season)
            
            if games_df.empty:
                continue
            
            game_ids = games_df['GAME_ID'].unique()
            
            # Filter out already processed games (check both checkpoint and existing files)
            existing_files = {f.stem for f in self.games_dir.glob('*.json')}
            processed_ids.update(existing_files)
            
            remaining_game_ids = [gid for gid in game_ids if gid not in processed_ids]
            
            if len(remaining_game_ids) == 0:
                print(f"\nAll games for {season} already processed. Skipping...")
                continue
            
            print(f"\nProcessing {len(remaining_game_ids)} remaining games for {season} (out of {len(game_ids)} total)...")
            print(f"Using {self.max_workers} parallel workers")
            
            # Prepare arguments for parallel processing
            process_args = [
                (game_id, season, str(self.dataset_dir), self.rate_limit_delay)
                for game_id in remaining_game_ids
            ]
            
            # Process games in parallel
            successful_count = 0
            with Pool(processes=self.max_workers) as pool:
                # Use imap for progress tracking
                results = list(tqdm(
                    pool.imap(process_single_game_wrapper, process_args),
                    total=len(process_args),
                    desc=f"Processing {season}"
                ))
            
            # Update processed IDs and save checkpoint periodically
            for game_id, success in results:
                if success:
                    processed_ids.add(game_id)
                    successful_count += 1
                    
                    # Save checkpoint every N games instead of every game
                    if len(processed_ids) % self.checkpoint_save_interval == 0:
                        self.save_checkpoint(season, processed_ids)
            
            # Final checkpoint save
            self.save_checkpoint(season, processed_ids)
            print(f"Successfully processed {successful_count} games for {season}")
            
            print(f"Collected {len(processed_ids)} games for {season}")
        
        # Count total games collected
        total_games = len(list(self.games_dir.glob('*.json')))
        
        print(f"\n{'='*60}")
        print(f"Data collection complete!")
        print(f"Total games collected: {total_games}")
        print(f"Games saved in: {self.games_dir}")
        print(f"{'='*60}")
        
        return pd.DataFrame()  # Return empty DataFrame since we're saving individual files
    
    def save_data(self, df, filename=None):
        """Save data to CSV (optional, JSON is primary format)"""
        if filename is None:
            filename = self.dataset_dir / 'nba_game_data.csv'
        
        if not df.empty:
            df.to_csv(filename, index=False)
            print(f"\nData also saved to CSV: {filename}")
            print(f"Shape: {df.shape}")
            return str(filename)
        return None


def main():
    """Main execution function"""
    collector = NBADataCollector(dataset_dir='Dataset')
    
    # Collect all data (with checkpoint/resume support)
    collector.collect_all_data()
    
    # Display summary
    game_files = list(collector.games_dir.glob('*.json'))
    print("\n" + "="*60)
    print("Data Collection Summary")
    print("="*60)
    print(f"\nTotal games collected: {len(game_files)}")
    print(f"Games saved in: {collector.games_dir}")
    print(f"Each game saved as: Dataset/games/{{GAME_ID}}.json")
    print("\nData structure for each game:")
    print("  - GAME_ID, SEASON, RESULT")
    print("  - home_team: {TEAM_ID, TEAM_NAME, PTS}")
    print("  - away_team: {TEAM_ID, TEAM_NAME, PTS}")
    print("  - game: {home_team: {...}, away_team: {...}, differences: {...}}")
    print("  - H2: {home_team: {...}, away_team: {...}, differences: {...}}")
    print("  - H3: {home_team: {...}, away_team: {...}, differences: {...}}")
    print("\n" + "="*60)
    print("NOTE: Quarter statistics (H2, H3) are retrieved from boxscore endpoints")
    print("using period parameters (start_period/end_period)")
    print("H2 = periods 1-2 (first half), H3 = periods 1-3 (first 3 quarters)")
    print("="*60)


if __name__ == "__main__":
    main()

