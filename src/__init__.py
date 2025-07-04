"""
QFTT Bootstrap Loop Simulation Package

A peer-review-grade simulation package for the Quantum Field Theory of Time (QFTT)
Bootstrap Loop using a minimal 3-qubit model (Clock, System, Environment).

© 2024 - MIT License
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@institution.edu"

from .qftt_simulator import QFTTSimulator
from . import qftt_analysis
from . import qftt_validation

__all__ = ["QFTTSimulator", "qftt_analysis", "qftt_validation"]