"""Demand forecasting and causal uplift engine."""

from .causal import run_causal
from .data import generate
from .forecast import BacktestConfig, backtest

__all__ = ["BacktestConfig", "backtest", "generate", "run_causal"]
__version__ = "0.1.0"
