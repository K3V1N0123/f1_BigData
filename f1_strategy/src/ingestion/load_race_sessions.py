import re
import unicodedata
from pathlib import Path

import fastf1
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
fastf1.Cache.enable_cache(str(PROJECT_ROOT / "data" / "cache"))


def normalize_event_name(event_name: str) -> str:
    normalized = unicodedata.normalize("NFKD", event_name).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^A-Za-z0-9]+", "_", normalized).strip("_")


def build_output_path(raw_root: Path, table_name: str, season: int, event_name: str) -> Path:
    normalized_event = normalize_event_name(event_name)
    return raw_root / table_name / f"season={season}" / f"event={normalized_event}" / "part.parquet"


def add_session_metadata(
    frame: pd.DataFrame,
    season: int,
    event_name: str,
    round_number: int,
    session_name: str,
    session_date: str,
) -> pd.DataFrame:
    result = frame.copy()
    result["season"] = season
    result["event_name"] = event_name
    result["round_number"] = round_number
    result["session_name"] = session_name
    result["session_date"] = session_date
    return result


def extract_session_tables(session) -> dict[str, pd.DataFrame]:
    return {
        "laps": session.laps.copy(),
        "weather": session.weather_data.copy(),
        "results": session.results.copy(),
        "race_control": session.race_control_messages.copy(),
    }


def write_table(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)


def ingest_loaded_session(session, raw_root: Path, season: int, event_name: str, round_number: int) -> None:
    session_name = str(session.name)
    session_date = str(session.date)[:10]
    tables = extract_session_tables(session)

    for table_name, frame in tables.items():
        enriched = add_session_metadata(
            frame=frame,
            season=season,
            event_name=event_name,
            round_number=round_number,
            session_name=session_name,
            session_date=session_date,
        )
        output_path = build_output_path(raw_root, table_name, season, event_name)
        write_table(enriched, output_path)


def ingest_seasons(
    seasons: list[int],
    raw_root: Path,
    get_event_schedule=fastf1.get_event_schedule,
    get_session=fastf1.get_session,
) -> dict[str, int]:
    summary = {"attempted": 0, "succeeded": 0, "failed": 0}

    for season in seasons:
        schedule = get_event_schedule(season)
        for _, event_row in schedule.iterrows():
            round_number = int(event_row["RoundNumber"])
            event_name = str(event_row["EventName"])
            summary["attempted"] += 1
            try:
                session = get_session(season, round_number, "R")
                session.load()
                ingest_loaded_session(session, raw_root, season, event_name, round_number)
                summary["succeeded"] += 1
            except Exception:
                summary["failed"] += 1

    return summary


def main() -> int:
    season_list = [2021, 2022, 2023, 2024]
    script_dir = Path(__file__).resolve().parent
    raw_root = script_dir.parent.parent / "data" / "raw"
    summary = ingest_seasons(seasons=season_list, raw_root=raw_root)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
 
