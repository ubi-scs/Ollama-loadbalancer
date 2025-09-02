import os
import json
from datetime import datetime

USAGE_STATS_PATH = 'usage_stats.json'

def log_usage(user, duration):
    """Append a usage record to the persistent usage stats file."""
    record = {
        "user": user,
        "timestamp": datetime.utcnow().isoformat(),
        "duration": duration  # in seconds
    }
    try:
        if not os.path.exists(USAGE_STATS_PATH):
            with open(USAGE_STATS_PATH, 'w') as f:
                json.dump([], f)
        with open(USAGE_STATS_PATH, 'r+') as f:
            try:
                data = json.load(f)
            except Exception:
                data = []
            data.append(record)
            f.seek(0)
            json.dump(data, f)
            f.truncate()
    except Exception as e:
        print(f"Error logging usage: {e}")

