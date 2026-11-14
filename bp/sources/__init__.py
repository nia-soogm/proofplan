"""Solvers for the Junction challenge data (data/jdata)."""
import os
from pathlib import Path
JDATA = Path(os.environ.get("BP_JDATA", Path(__file__).resolve().parent.parent.parent / "data" / "jdata"))  # see DATA.md
CACHE = Path(__file__).resolve().parent.parent.parent / "data" / "jcache"
