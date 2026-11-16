"""Child-process entry point used by runner.run_tests_isolated.  Usage: python -m harness.worker job.json out.json"""
import json
import sys
from pathlib import Path

from .runner import run_tests
from .testcase import TestCase


def main() -> int:
    job = json.loads(Path(sys.argv[1]).read_text())
    tests = [TestCase.from_dict(t) for t in job["tests"]]
    res = run_tests(job["op"], job["ptx"], job["kernel"], tuple(job["block"]), tests)
    Path(sys.argv[2]).write_text(json.dumps(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
