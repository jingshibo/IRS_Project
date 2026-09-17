from __future__ import annotations

import os
from pathlib import Path


CLASS_ORDER = ("LOW", "TARGET", "HIGH")
SIGNAL_SEGMENTS = ((0, 1000), (1800, 3500))
DOWNSAMPLE_STEP = 5
RANDOM_SEED = 42

RAW_EXAMPLES_PER_CLASS = 8
RAW_SUBPLOTS_PER_CLASS = 4
FEATURE_TABLE_ROWS_PER_CLASS = 2

# Fixed mystery samples for the interactive game. These IDs use the original
# dataset label plus the local sample index shown in the generated HTML payload.
FIXED_MYSTERY_SAMPLE_IDS = (
    "LOW-211",
    "LOW-281",
    "LOW-391",
    "LOW-645",
    "LOW-512",
    "LOW-503",
    "TARGET-543",
    "TARGET-481",
    "TARGET-101",
    "TARGET-606",
    "TARGET-557",
    "TARGET-284",
    "HIGH-489",
    "HIGH-345",
    "HIGH-1",
    "HIGH-435",
    "HIGH-366",
    "HIGH-413",
)

DEFAULT_DATA_CANDIDATES = (
    Path(os.environ["IRS_SCHOOL_VISIT_DATA_PATH"])
    if os.environ.get("IRS_SCHOOL_VISIT_DATA_PATH")
    else None,
    Path("/home/shibojing/data/Practice/Stage3a_all_mixed.xlsx"),
    Path.home() / "data" / "Practice" / "Stage3a_all_mixed.xlsx",
    Path.home() / "Downloads" / "Stage3a_all_mixed.xlsx",
)
