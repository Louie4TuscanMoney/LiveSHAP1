# Railway Deployment Guide

## Quick Start - Git Commands

Run these commands in your terminal:

```bash
cd /Users/embrace/Desktop/SHAP && \
git init && \
git remote add origin https://github.com/Louie4TuscanMoney/LiveSHAP1.git && \
git add . && \
git commit -m 'Initial commit: NBA Live Prediction API with Railway deployment' && \
git branch -M main && \
git push -u origin main
```

## Step-by-Step Instructions

### 1. Initialize Git Repository
```bash
cd /Users/embrace/Desktop/SHAP
git init
```

### 2. Add Remote Repository
```bash
git remote add origin https://github.com/Louie4TuscanMoney/LiveSHAP1.git
```

### 3. Stage All Files
```bash
git add .
```

### 4. Commit Changes
```bash
git commit -m 'Initial commit: NBA Live Prediction API with Railway deployment'
```

### 5. Set Main Branch and Push
```bash
git branch -M main
git push -u origin main
```

## Railway Deployment

### 1. Connect GitHub Repository
1. Go to [Railway.app](https://railway.app)
2. Click "New Project"
3. Select "Deploy from GitHub repo"
4. Choose `LiveSHAP1` repository

### 2. Configure Environment
Railway will automatically detect:
- **Build Command**: Auto-detected from `Procfile`
- **Start Command**: `gunicorn api_server:app --bind 0.0.0.0:$PORT`
- **Port**: Automatically set via `$PORT` environment variable

### 3. Deploy
Railway will automatically:
- Install dependencies from `requirements.txt`
- Build the application
- Start the API server

## API Endpoints

### Health Check
```
GET /
```

### Start Monitoring
```
POST /api/run
```

### Get All Games (Grouped by game_id)
```
GET /api/games
```
Returns games sorted by latest prediction timestamp, perfect for frontend sorting.

### Get Specific Game
```
GET /api/games/{game_id}
```

### Get All Predictions (Flat List)
```
GET /api/predictions
```

### Get Statistics
```
GET /api/stats
```

### Get Monitoring Status
```
GET /api/status
```

### Stop Monitoring
```
POST /api/stop
```

## Frontend Integration

### Example: Fetch All Games
```javascript
fetch('https://your-app.railway.app/api/games')
  .then(res => res.json())
  .then(data => {
    // data.games is an array of games, each with:
    // - game_id
    // - home_team, away_team
    // - predictions[] (array of H2/H3 predictions)
    // - latest_prediction (most recent)
    // - has_outcome (boolean)
    console.log(data.games);
  });
```

### Example: Fetch Specific Game
```javascript
fetch('https://your-app.railway.app/api/games/0022500358')
  .then(res => res.json())
  .then(data => {
    // data.predictions contains all predictions for this game
    console.log(data.predictions);
  });
```

### Example: Start Monitoring
```javascript
fetch('https://your-app.railway.app/api/run', {
  method: 'POST'
})
  .then(res => res.json())
  .then(data => {
    console.log('Monitoring started:', data);
  });
```

## Response Format

### `/api/games` Response Structure
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
          "predicted_prob": 0.233,
          "ci_lower": 0.147,
          "ci_upper": 0.319,
          "home_score_at_prediction": 59,
          "away_score_at_prediction": 77,
          "predicted_outcome": 0,
          "actual_outcome": null
        }
      ],
      "latest_prediction": {
        "period_type": "H2",
        "timestamp": "2025-12-07T13:45:53",
        "predicted_prob_percent": 23.3,
        ...
      },
      "has_outcome": false
    }
  ],
  "total_games": 1,
  "timestamp": "2025-12-07T14:00:00"
}
```

## Important Notes

1. **Models Required**: Make sure `3modeltraining/models/xgb_model_H2.pkl` and `xgb_model_H3.pkl` are committed to the repository (they're currently in `.gitignore` - you may need to remove them from gitignore or commit them separately).

2. **Predictions Storage**: Predictions are stored in `4liveprediction/predictions/predictions.json` and `predictions.csv`. These files persist across deployments on Railway.

3. **Monitoring**: The monitoring script runs in a background thread when you call `/api/run`. It checks for live games every 10 seconds.

4. **CORS**: CORS is enabled for all origins, so your frontend can call the API from any domain.

## Troubleshooting

### Models Not Found
If you get "Model not found" errors:
1. Check that model files exist in `3modeltraining/models/`
2. Remove `*.pkl` from `.gitignore` if needed
3. Commit and push model files

### Monitoring Not Starting
- Check `/api/status` to see if monitoring is active
- Check Railway logs for errors
- Ensure models are available

### API Not Responding
- Check Railway deployment logs
- Verify `PORT` environment variable is set (Railway does this automatically)
- Check that `gunicorn` is installed (included in requirements.txt)

