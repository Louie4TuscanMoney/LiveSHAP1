# Auto-Monitoring System for SHAP

## Overview

The auto-monitoring system automatically starts SHAP monitoring when NBA games are within 2 minutes of starting and stops monitoring when games conclude.

## How It Works

1. **Auto-Start**: Checks NBA scoreboard every 30 seconds
   - Detects games scheduled to start within 2 minutes
   - Automatically starts monitoring via API endpoint
   - Tracks which games are being monitored

2. **Auto-Stop**: The existing monitor automatically stops tracking games when they finish
   - Games marked as "Final" are removed from tracking
   - Monitoring continues for other live games

## Features

- ✅ Auto-starts monitoring 2 minutes before game start
- ✅ Per-game tracking (knows which games are being monitored)
- ✅ Auto-stops when games conclude
- ✅ Runs continuously in background
- ✅ Integrates with existing monitoring system

## API Endpoints

### Check Status
```bash
GET /api/status
```

Returns:
```json
{
  "monitoring": {
    "active": true,
    "started_at": "2025-12-08T...",
    "last_check": "..."
  },
  "auto_monitor": {
    "active": true,
    "active_games": 2,
    "games": [
      {
        "game_id": "0022500363",
        "started_at": "2025-12-08T...",
        "game_info": {
          "home_team": "Lakers",
          "away_team": "Warriors"
        }
      }
    ]
  }
}
```

## Configuration

The auto-monitor starts automatically when the API server starts.

To disable auto-monitoring, set environment variable:
```bash
DISABLE_AUTO_MONITOR=true
```

## Manual Control

You can still manually start/stop monitoring:
```bash
# Start monitoring
POST /api/run

# Stop monitoring  
POST /api/stop
```

## Monitoring Window

- **Start Window**: 2 minutes before game start time
- **Check Interval**: Every 30 seconds
- **Stop**: When game status is "Final"

## Notes

- The existing `monitor_live_games.py` monitors all live games simultaneously
- Auto-monitor ensures monitoring is active when games are about to start
- Each game is tracked individually, but monitoring runs as a single process
- Games are automatically removed from tracking when they finish

