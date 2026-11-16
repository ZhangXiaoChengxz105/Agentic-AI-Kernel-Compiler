"""Task 1.4: PyTorch reference implementations and input generators (see ops.py, inputs.py)."""
from .inputs import DISTS, make_tensor
from .ops import OPS, OpSpec

__all__ = ["DISTS", "OPS", "OpSpec", "make_tensor"]
