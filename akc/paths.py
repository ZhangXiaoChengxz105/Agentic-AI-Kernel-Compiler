"""Filesystem locations of the project (everything lives inside the workspace root)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKLOADS = ROOT / "workloads"
INCLUDE_DIR = WORKLOADS / "common"            # common.cuh (EXP macro)
RESULTS = ROOT / "results"
BUILD = ROOT / "build"
LOGS = ROOT / "logs"
THIRD_PARTY = ROOT / "third_party"
VOLTA_BIN = THIRD_PARTY / "volta" / "target" / "release" / "volta"
Z3_LIB_DIR = ROOT / "tools" / "z3" / "z3" / "lib"
