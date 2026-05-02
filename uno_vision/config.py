from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
SUBMISSION_DIR = PROJECT_ROOT / "submissions"

CONDITION_TOKENS = ["center", "top", "right", "bottom", "left"]
CONDITION_TO_INDEX = {name: idx for idx, name in enumerate(CONDITION_TOKENS)}

PLAYERS = ["p1", "p2", "p3", "p4"]
PLAYER_TO_INDEX = {name: idx for idx, name in enumerate(PLAYERS)}

PLAYER_TO_CONDITION = {
    "p1": "bottom",
    "p2": "left",
    "p3": "top",
    "p4": "right",
}

CONDITION_TO_PLAYER = {condition: player for player, condition in PLAYER_TO_CONDITION.items()}

PARAMETER_LIMIT = 12_000_000

