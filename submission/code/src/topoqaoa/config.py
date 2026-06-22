"""YAML experiment configuration."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import yaml


@dataclass
class Config:
    name: str = "smoke"
    seed: int = 0
    families: List[str] = field(default_factory=lambda: [
        "erdos_renyi", "regular", "barabasi_albert",
        "watts_strogatz", "grid", "stochastic_block",
    ])
    n_per_family: int = 3
    sizes: List[int] = field(default_factory=lambda: [6, 8])
    budget: int = 12
    target_ratio: float = 0.95

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        data = yaml.safe_load(Path(path).read_text()) or {}
        return cls(**data)
