"""Child process for perf.measure.  Usage: python -m akc.perf.worker job.json out.json"""
import json
import sys
from pathlib import Path

from .measure import _child

if __name__ == "__main__":
    Path(sys.argv[2]).write_text(json.dumps(_child(json.loads(Path(sys.argv[1]).read_text())), default=str))
