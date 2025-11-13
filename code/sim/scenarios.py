"""Scenario definitions for simulation experiments.
Centralizes scenario configs so both the runner and exporters can import.
"""
from typing import List, Dict


def get_scenarios() -> List[Dict]:
    return [
        {"name": "baseline",   "latency_steps": 0, "dropout_prob": 0.0, "noise_std": 0.0},
        {"name": "latency_3",  "latency_steps": 3, "dropout_prob": 0.0, "noise_std": 0.0},
        {"name": "dropout_0.2","latency_steps": 0, "dropout_prob": 0.2, "noise_std": 0.0},
        {"name": "noise_0.02", "latency_steps": 0, "dropout_prob": 0.0, "noise_std": 0.02},
        {"name": "combo",      "latency_steps": 3, "dropout_prob": 0.2, "noise_std": 0.02},
    ]
