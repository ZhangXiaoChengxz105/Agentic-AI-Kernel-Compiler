"""Child process for runners.gpu.run_tests_isolated.  Usage: python -m akc.validation.runners.gpu_worker job.json out.json"""
import json
import sys
from pathlib import Path

from ...task.testcase import TestCase
from .gpu import run_tests


def main() -> int:
    job = json.loads(Path(sys.argv[1]).read_text())
    tests = [TestCase.from_dict(t) for t in job["tests"]]
    res = run_tests(job["workload"], job["lang"], job["kernel_file"], job["entry"], tuple(job["block"]), tests)
    Path(sys.argv[2]).write_text(json.dumps(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
