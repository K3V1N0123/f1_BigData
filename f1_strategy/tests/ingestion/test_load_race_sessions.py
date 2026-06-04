import sys
from pathlib import Path

import pandas as pd
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.load_race_sessions import add_session_metadata
from src.ingestion.load_race_sessions import extract_session_tables
from src.ingestion.load_race_sessions import ingest_seasons
from src.ingestion.load_race_sessions import ingest_loaded_session
from src.ingestion.load_race_sessions import parse_args
from src.ingestion.load_race_sessions import normalize_event_name
from src.ingestion.load_race_sessions import build_output_path
from src.ingestion.load_race_sessions import write_table


def test_normalize_event_name_replaces_spaces_and_strips_symbols():
    assert normalize_event_name("Bahrain Grand Prix") == "Bahrain_Grand_Prix"
    assert normalize_event_name("Sao Paulo Grand Prix") == "Sao_Paulo_Grand_Prix"


def test_normalize_event_name_removes_accents():
    assert normalize_event_name("São Paulo Grand Prix") == "Sao_Paulo_Grand_Prix"


def test_build_output_path_uses_table_season_and_event_partitions():
    path = build_output_path(
        raw_root=Path("data/raw"),
        table_name="laps",
        season=2023,
        event_name="São Paulo Grand Prix",
    )

    assert path == Path("data/raw/laps/season=2023/event=Sao_Paulo_Grand_Prix/part.parquet")


def test_add_session_metadata_appends_required_columns():
    frame = pd.DataFrame({"Driver": ["VER"]})

    result = add_session_metadata(
        frame,
        season=2023,
        event_name="Bahrain Grand Prix",
        round_number=1,
        session_name="Race",
        session_date="2023-03-05",
    )

    assert result["Driver"].tolist() == ["VER"]
    assert result["season"].tolist() == [2023]
    assert result["event_name"].tolist() == ["Bahrain Grand Prix"]
    assert result["round_number"].tolist() == [1]
    assert result["session_name"].tolist() == ["Race"]
    assert result["session_date"].tolist() == ["2023-03-05"]


def test_extract_session_tables_returns_expected_frames():
    session = SimpleNamespace(
        laps=pd.DataFrame({"Driver": ["VER"]}),
        weather_data=pd.DataFrame({"AirTemp": [31.0]}),
        results=pd.DataFrame({"Abbreviation": ["VER"]}),
        race_control_messages=pd.DataFrame({"Message": ["GREEN LIGHT"]}),
    )

    tables = extract_session_tables(session)

    assert set(tables.keys()) == {"laps", "weather", "results", "race_control"}
    assert tables["laps"].iloc[0]["Driver"] == "VER"
    assert tables["weather"].iloc[0]["AirTemp"] == 31.0
    assert tables["results"].iloc[0]["Abbreviation"] == "VER"
    assert tables["race_control"].iloc[0]["Message"] == "GREEN LIGHT"


def test_write_table_creates_partitioned_parquet_file(tmp_path: Path):
    frame = pd.DataFrame({"Driver": ["VER"]})
    output_path = tmp_path / "data/raw/laps/season=2023/event=Bahrain_Grand_Prix/part.parquet"

    write_table(frame, output_path)

    assert output_path.exists()
    reloaded = pd.read_parquet(output_path)
    assert reloaded["Driver"].tolist() == ["VER"]


def test_ingest_loaded_session_writes_all_expected_tables(tmp_path: Path):
    session = SimpleNamespace(
        name="Race",
        date="2023-03-05",
        laps=pd.DataFrame({"Driver": ["VER"]}),
        weather_data=pd.DataFrame({"AirTemp": [31.0]}),
        results=pd.DataFrame({"Abbreviation": ["VER"]}),
        race_control_messages=pd.DataFrame({"Message": ["GREEN LIGHT"]}),
    )

    ingest_loaded_session(
        session=session,
        raw_root=tmp_path / "data/raw",
        season=2023,
        event_name="Bahrain Grand Prix",
        round_number=1,
    )

    assert (tmp_path / "data/raw/laps/season=2023/event=Bahrain_Grand_Prix/part.parquet").exists()
    assert (tmp_path / "data/raw/weather/season=2023/event=Bahrain_Grand_Prix/part.parquet").exists()
    assert (tmp_path / "data/raw/results/season=2023/event=Bahrain_Grand_Prix/part.parquet").exists()
    assert (tmp_path / "data/raw/race_control/season=2023/event=Bahrain_Grand_Prix/part.parquet").exists()

    laps = pd.read_parquet(tmp_path / "data/raw/laps/season=2023/event=Bahrain_Grand_Prix/part.parquet")
    assert laps["season"].tolist() == [2023]
    assert laps["event_name"].tolist() == ["Bahrain Grand Prix"]
    assert laps["round_number"].tolist() == [1]
    assert laps["session_name"].tolist() == ["Race"]
    assert laps["session_date"].tolist() == ["2023-03-05"]


class FakeSession:
    def __init__(self, name: str, date: str):
        self.name = name
        self.date = date
        self.laps = pd.DataFrame({"Driver": ["VER"]})
        self.weather_data = pd.DataFrame({"AirTemp": [31.0]})
        self.results = pd.DataFrame({"Abbreviation": ["VER"]})
        self.race_control_messages = pd.DataFrame({"Message": ["GREEN LIGHT"]})

    def load(self):
        return None


def test_ingest_seasons_continues_when_one_session_fails(tmp_path: Path):
    schedule = pd.DataFrame([
        {"RoundNumber": 1, "EventName": "Bahrain Grand Prix"},
        {"RoundNumber": 2, "EventName": "Saudi Arabian Grand Prix"},
    ])
    calls = []

    def fake_get_schedule(season: int):
        assert season == 2023
        return schedule

    def fake_get_session(season: int, round_number: int, session_code: str):
        calls.append((season, round_number, session_code))
        if round_number == 2:
            raise RuntimeError("boom")
        return FakeSession("Race", "2023-03-05")

    summary = ingest_seasons(
        seasons=[2023],
        raw_root=tmp_path / "data/raw",
        get_event_schedule=fake_get_schedule,
        get_session=fake_get_session,
    )

    assert summary == {"attempted": 2, "succeeded": 1, "failed": 1}
    assert calls == [(2023, 1, "R"), (2023, 2, "R")]
    assert (tmp_path / "data/raw/laps/season=2023/event=Bahrain_Grand_Prix/part.parquet").exists()


def test_parse_args_defaults_to_2021_through_2024():
    args = parse_args([])

    assert args.start_year == 2021
    assert args.end_year == 2024


def test_parse_args_rejects_removed_data_root_option():
    try:
        parse_args(["--data-root", "somewhere"])
    except SystemExit as error:
        assert error.code == 2
    else:
        raise AssertionError("--data-root should no longer be accepted")
