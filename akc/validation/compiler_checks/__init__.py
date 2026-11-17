"""The compiler-checks gate (D2): tirx-lite's own checks + sanitizer feedback, run before the tests and VOLTA.

    validator.py    CompilerChecksValidator ("compiler_checks"): the gate of configurations A / B / C
    tirx_static.py  static checks in a child process (trace + IR contract, TIRx analyses, sm_80 compile) and the
                    CPU-only NumSim racecheck / synccheck stage; gate_shapes / numsim_shape
    sanitizer.py    compute-sanitizer memcheck / racecheck / initcheck / synccheck on the generated CUDA (GPU),
                    log parser
    launcher.py     the torch-free driver-API launcher the sanitizer tools wrap (run by path)
"""
from .validator import CompilerChecksValidator

__all__ = ["CompilerChecksValidator"]
