#!/usr/bin/env python3
"""
Flask API Server for NBA Live Prediction System
Deployed on Railway for frontend integration
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import threading
import time
import json
from pathlib import Path
from datetime import datetime
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import prediction storage dynamically (module name starts with digit)
import importlib.util
prediction_storage_path = Path('4liveprediction/prediction_storage.py')
prediction_spec = importlib.util.spec_from_file_location("prediction_storage", prediction_storage_path)
prediction_storage_module = importlib.util.module_from_spec(prediction_spec)
prediction_spec.loader.exec_module(prediction_storage_module)
load_predictions = prediction_storage_module.load_predictions
get_predictions_with_outcomes = prediction_storage_module.get_predictions_with_outcomes

# Import monitoring function dynamically
monitor_path = Path('4liveprediction/monitor_live_games.py')
monitor_spec = importlib.util.spec_from_file_location("monitor", monitor_path)
monitor_module = importlib.util.module_from_spec(monitor_spec)
monitor_spec.loader.exec_module(monitor_module)
monitor_main = monitor_module.main

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Global state
monitoring_thread = None
monitoring_active = False
auto_monitor_thread = None
auto_monitor_active = False
monitoring_status = {
    'active': False,
    'started_at': None,
    'last_check': None,
    'games_monitored': 0
}

# Per-game monitoring tracking
active_game_monitors = {}  # {game_id: {'thread': thread, 'started_at': str}}
monitoring_lock = threading.Lock()

# Storage paths
PREDICTIONS_JSON = Path('4liveprediction/predictions/predictions.json')
PREDICTIONS_CSV = Path('4liveprediction/predictions/predictions.csv')

def run_monitoring():
    """Run the monitoring script in a separate thread."""
    global monitoring_active, monitoring_status
    monitoring_active = True
    monitoring_status['active'] = True
    monitoring_status['started_at'] = datetime.now().isoformat()
    
    try:
        # Import and run the monitor
        monitor_main()
    except Exception as e:
        print(f"Error in monitoring: {e}")
        monitoring_active = False
        monitoring_status['active'] = False

@app.route('/')
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'service': 'NBA Live Prediction API',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get monitoring status."""
    with monitoring_lock:
        active_games = len(active_game_monitors)
        game_list = [
            {
                'game_id': game_id,
                'started_at': info.get('started_at'),
                'game_info': info.get('game_info', {})
            }
            for game_id, info in active_game_monitors.items()
        ]
    
    return jsonify({
        'monitoring': {
            'active': monitoring_status['active'],
            'started_at': monitoring_status['started_at'],
            'last_check': monitoring_status['last_check']
        },
        'auto_monitor': {
            'active': auto_monitor_active,
            'active_games': active_games,
            'games': game_list
        },
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/run', methods=['POST'])
def start_monitoring():
    """Start the live game monitoring script."""
    global monitoring_thread, monitoring_active, monitoring_status
    
    if monitoring_active:
        return jsonify({
            'status': 'already_running',
            'message': 'Monitoring is already active',
            'started_at': monitoring_status['started_at']
        }), 200
    
    try:
        # Start monitoring in background thread
        monitoring_thread = threading.Thread(target=run_monitoring, daemon=True)
        monitoring_thread.start()
        
        return jsonify({
            'status': 'started',
            'message': 'Live game monitoring started',
            'started_at': monitoring_status['started_at']
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Failed to start monitoring: {str(e)}'
        }), 500

@app.route('/api/stop', methods=['POST'])
def stop_monitoring():
    """Stop the live game monitoring script."""
    global monitoring_active, monitoring_status
    
    if not monitoring_active:
        return jsonify({
            'status': 'not_running',
            'message': 'Monitoring is not currently active'
        }), 200
    
    monitoring_active = False
    monitoring_status['active'] = False
    
    return jsonify({
        'status': 'stopped',
        'message': 'Live game monitoring stopped',
        'stopped_at': datetime.now().isoformat()
    }), 200

@app.route('/api/games', methods=['GET'])
def get_all_games():
    """Get all games with their predictions, grouped by game_id."""
    try:
        if not PREDICTIONS_JSON.exists():
            return jsonify({
                'games': {},
                'total_games': 0,
                'message': 'No predictions found'
            }), 200
        
        with open(PREDICTIONS_JSON, 'r') as f:
            predictions = json.load(f)
        
        # Group predictions by game_id
        games_dict = {}
        for pred in predictions:
            game_id = pred['game_id']
            
            if game_id not in games_dict:
                games_dict[game_id] = {
                    'game_id': game_id,
                    'home_team': pred.get('home_team', ''),
                    'away_team': pred.get('away_team', ''),
                    'predictions': [],
                    'latest_prediction': None,
                    'has_outcome': pred.get('actual_outcome') is not None
                }
            
            # Add prediction
            prediction_data = {
                'period_type': pred.get('period_type'),
                'timestamp': pred.get('timestamp'),
                'predicted_prob_percent': pred.get('predicted_prob_percent'),
                'predicted_prob': pred.get('predicted_prob'),
                'ci_lower': pred.get('ci_lower'),
                'ci_upper': pred.get('ci_upper'),
                'home_score_at_prediction': pred.get('home_score_at_prediction'),
                'away_score_at_prediction': pred.get('away_score_at_prediction'),
                'predicted_outcome': pred.get('predicted_outcome'),
                'actual_outcome': pred.get('actual_outcome')
            }
            
            games_dict[game_id]['predictions'].append(prediction_data)
            
            # Update latest prediction
            if games_dict[game_id]['latest_prediction'] is None:
                games_dict[game_id]['latest_prediction'] = prediction_data
            else:
                latest_time = datetime.fromisoformat(games_dict[game_id]['latest_prediction']['timestamp'])
                current_time = datetime.fromisoformat(pred['timestamp'])
                if current_time > latest_time:
                    games_dict[game_id]['latest_prediction'] = prediction_data
        
        # Convert to list and sort by latest prediction timestamp
        games_list = list(games_dict.values())
        games_list.sort(
            key=lambda x: x['latest_prediction']['timestamp'] if x['latest_prediction'] else '',
            reverse=True
        )
        
        return jsonify({
            'games': games_list,
            'total_games': len(games_list),
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'message': 'Failed to load predictions'
        }), 500

@app.route('/api/games/<game_id>', methods=['GET'])
def get_game_predictions(game_id):
    """Get all predictions for a specific game."""
    try:
        if not PREDICTIONS_JSON.exists():
            return jsonify({
                'game_id': game_id,
                'predictions': [],
                'message': 'No predictions found for this game'
            }), 200
        
        with open(PREDICTIONS_JSON, 'r') as f:
            predictions = json.load(f)
        
        # Filter predictions for this game
        game_predictions = [p for p in predictions if p['game_id'] == game_id]
        
        if not game_predictions:
            return jsonify({
                'game_id': game_id,
                'predictions': [],
                'message': 'No predictions found for this game'
            }), 200
        
        # Sort by timestamp (oldest first)
        game_predictions.sort(key=lambda x: x.get('timestamp', ''))
        
        # Extract game info from first prediction
        first_pred = game_predictions[0]
        game_info = {
            'game_id': game_id,
            'home_team': first_pred.get('home_team', ''),
            'away_team': first_pred.get('away_team', ''),
            'predictions': game_predictions,
            'has_outcome': any(p.get('actual_outcome') is not None for p in game_predictions)
        }
        
        return jsonify(game_info), 200
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'message': 'Failed to load game predictions'
        }), 500

@app.route('/api/predictions', methods=['GET'])
def get_all_predictions():
    """Get all predictions (flat list, not grouped by game)."""
    try:
        if not PREDICTIONS_JSON.exists():
            return jsonify({
                'predictions': [],
                'total': 0
            }), 200
        
        with open(PREDICTIONS_JSON, 'r') as f:
            predictions = json.load(f)
        
        # Sort by timestamp (newest first)
        predictions.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        return jsonify({
            'predictions': predictions,
            'total': len(predictions),
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'message': 'Failed to load predictions'
        }), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get overall prediction statistics."""
    try:
        if not PREDICTIONS_JSON.exists():
            return jsonify({
                'total_predictions': 0,
                'total_games': 0,
                'predictions_with_outcomes': 0
            }), 200
        
        with open(PREDICTIONS_JSON, 'r') as f:
            predictions = json.load(f)
        
        # Calculate stats
        total_predictions = len(predictions)
        unique_games = len(set(p['game_id'] for p in predictions))
        predictions_with_outcomes = sum(1 for p in predictions if p.get('actual_outcome') is not None)
        
        # Count by period type
        h2_count = sum(1 for p in predictions if p.get('period_type') == 'H2')
        h3_count = sum(1 for p in predictions if p.get('period_type') == 'H3')
        
        return jsonify({
            'total_predictions': total_predictions,
            'total_games': unique_games,
            'predictions_with_outcomes': predictions_with_outcomes,
            'h2_predictions': h2_count,
            'h3_predictions': h3_count,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'message': 'Failed to calculate stats'
        }), 500

def start_auto_monitor():
    """Start the auto-monitoring system in background."""
    global auto_monitor_thread, auto_monitor_active, monitoring_active
    
    if auto_monitor_active:
        return
    
    try:
        # Import auto_monitor dynamically
        auto_monitor_path = Path(__file__).parent / 'auto_monitor.py'
        if auto_monitor_path.exists():
            import importlib.util
            auto_monitor_spec = importlib.util.spec_from_file_location("auto_monitor", auto_monitor_path)
            auto_monitor_module = importlib.util.module_from_spec(auto_monitor_spec)
            
            # Pass the start_monitoring function to auto_monitor
            auto_monitor_module.start_monitoring_func = lambda: start_monitoring() if not monitoring_active else None
            auto_monitor_module.monitoring_active_ref = lambda: monitoring_active
            
            auto_monitor_spec.loader.exec_module(auto_monitor_module)
            
            auto_monitor_active = True
            auto_monitor_thread = threading.Thread(
                target=auto_monitor_module.auto_monitor_loop,
                daemon=True,
                name="AutoMonitor"
            )
            auto_monitor_thread.start()
            print("✅ Auto-monitoring started")
        else:
            print("⚠️  auto_monitor.py not found - auto-monitoring disabled")
    except Exception as e:
        print(f"⚠️  Error starting auto-monitor: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    # Start auto-monitoring in background
    start_auto_monitor()
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

