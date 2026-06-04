from pathlib import Path

import fastf1


PROJECT_ROOT = Path(__file__).resolve().parents[2]
fastf1.Cache.enable_cache(str(PROJECT_ROOT / 'data' / 'cache'))
session = fastf1.get_session(2023, 'Bahrain', 'R')
session.load()
laps = session.laps          # Main laps DataFrame
weather = session.weather_data  # Weather DataFrame
print(laps.head())
print(weather.head())
