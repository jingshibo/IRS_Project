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

DEFAULT_DATA_CANDIDATES = (
    Path(os.environ["IRS_SCHOOL_VISIT_DATA_PATH"])
    if os.environ.get("IRS_SCHOOL_VISIT_DATA_PATH")
    else None,
    Path("/home/shibojing/data/Practice/Stage3a_all_mixed.xlsx"),
    Path.home() / "data" / "Practice" / "Stage3a_all_mixed.xlsx",
    Path.home() / "Downloads" / "Stage3a_all_mixed.xlsx",
)
