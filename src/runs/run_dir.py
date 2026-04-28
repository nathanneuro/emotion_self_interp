"""Per-run timestamped output directories with config.json snapshot."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = PROJECT_ROOT / "outputs"


def make_run_dir(name: str, config: dict[str, Any]) -> Path:
    """Create outputs/<name>_<YYYYMMDD_HHMM>/ and write config.json into it."""
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    safe = name.replace("/", "_").replace(" ", "_")
    rd = OUTPUTS / f"{safe}_{ts}"
    rd.mkdir(parents=True, exist_ok=False)
    (rd / "config.json").write_text(json.dumps(config, indent=2, default=str))
    return rd
