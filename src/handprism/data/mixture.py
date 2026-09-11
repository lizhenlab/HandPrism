"""Homogeneous ARCTIC/HOT3D batch sampling with explicit mixture weights."""

from __future__ import annotations

import random
import math
from collections.abc import Mapping, Sequence
from typing import Any
from .policy import require_dataset


DEFAULT_DATASET_WEIGHTS = {"arctic": 0.4375, "hot3d": 0.5625}


class DatasetMixture:
    """Choose one dataset per batch; never stack heterogeneous resolutions."""

    def __init__(
        self,
        datasets: Mapping[str, Sequence[Any]],
        weights: Mapping[str, float] | None = None,
        seed: int = 260820308,
    ) -> None:
        if weights is None:
            weights = DEFAULT_DATASET_WEIGHTS.copy()
        for name in set(datasets) | set(weights):
            require_dataset(name)
        if any(not math.isfinite(weight) or weight <= 0 for weight in weights.values()):
            raise ValueError("dataset weights must be finite and positive")
        missing = set(weights) - set(datasets)
        if missing:
            raise ValueError(f"missing configured datasets: {sorted(missing)}")
        if abs(sum(weights.values()) - 1.0) > 1e-8:
            raise ValueError("dataset weights must sum to one")
        if any(len(datasets[name]) == 0 for name in weights):
            raise ValueError("every configured dataset must be non-empty")
        self.datasets = datasets
        self.names = tuple(weights)
        self.weights = tuple(weights[name] for name in self.names)
        self.rng = random.Random(seed)

    def sample_batch(self, batch_size: int) -> tuple[str, list[Any]]:
        name = self.rng.choices(self.names, weights=self.weights, k=1)[0]
        dataset = self.datasets[name]
        return name, [dataset[self.rng.randrange(len(dataset))] for _ in range(batch_size)]
