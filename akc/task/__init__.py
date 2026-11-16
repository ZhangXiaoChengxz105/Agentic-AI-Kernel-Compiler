"""What the agent must solve: workloads (operator + manifest), references, inputs, test cases."""
from .testcase import TestCase
from .workload import Variant, Workload, load_workload, list_workloads

__all__ = ["TestCase", "Variant", "Workload", "load_workload", "list_workloads"]
