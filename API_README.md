# API Server for Railway Deployment

## API Endpoints

### Health Check
- `GET /` - Health check endpoint

### Monitoring Control
- `POST /api/run` - Start live game monitoring
- `POST /api/stop` - Stop live game monitoring
- `GET /api/status` - Get monitoring status

### Game Data
- `GET /api/games` - Get all games with predictions (grouped by game_id)
- `GET /api/games/<game_id>` - Get all predictions for a specific game
- `GET /api/predictions` - Get all predictions (flat list)
- `GET /api/stats` - Get overall prediction statistics

## Example Usage

### Start Monitoring
```bash
curl -X POST https://your-railway-app.railway.app/api/run
```

### Get All Games
```bash
curl https://your-railway-app.railway.app/api/games
```

### Get Specific Game
```bash
curl https://your-railway-app.railway.app/api/games/0022500358
```

## Response Format

### `/api/games` Response
```json
{
  "games": [
    {
      "game_id": "0022500358",
      "home_team": "Raptors",
      "away_team": "Celtics",
      "predictions": [
        {
          "period_type": "H2",
          "timestamp": "2025-12-07T13:45:53",
          "predicted_prob_percent": 23.3,
          "ci_lower": 0.147,
          "ci_upper": 0.319,
          "predicted_outcome": 0,
          "actual_outcome": null
        }
      ],
      "latest_prediction": {...},
      "has_outcome": false
    }
  ],
  "total_games": 1
}
```

## Deployment

The API server runs on Railway using Gunicorn. The `Procfile` and `railway.json` are configured for automatic deployment.

