"""Combinatorics for systematic experiments over analysis settings.

Every analysis setting is a finite set of choices S₁, S₂, …, S_k. A complete
configuration is one element of the Cartesian product S₁ × S₂ × … × S_k, so
by the rule of product there are |S₁|·|S₂|·…·|S_k| configurations. Some are
ruled out by constraints (counting with restrictions), and running only m of
the n valid ones can be done in C(n, m) different ways.

Because preprocessing only depends on the preprocessing parameters, the
number of *distinct* preprocessing runs is the size of the projection of the
configuration set onto those parameters — usually far fewer than the total.
"""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Sequence


@dataclass
class Parameter:
    key: str
    label: str
    values: list[Any]
    group: str  # "preprocess" | "threshold" | "bls" | "matching"
    unit: str = ""

    def fmt(self, v: Any) -> str:
        return "off" if v is None else f"{v:g}{self.unit}" if isinstance(v, (int, float)) else str(v)


@dataclass
class Constraint:
    description: str
    check: Callable[[dict[str, Any]], bool]


@dataclass
class ParameterSpace:
    parameters: list[Parameter]
    constraints: list[Constraint] = field(default_factory=list)

    def validate(self) -> None:
        keys = [p.key for p in self.parameters]
        if len(set(keys)) != len(keys):
            raise ValueError("parameter keys must be unique")
        for p in self.parameters:
            if not p.values:
                raise ValueError(f"parameter '{p.label}' needs at least one value")
            if len(set(map(repr, p.values))) != len(p.values):
                raise ValueError(f"parameter '{p.label}' has duplicate values (a set has distinct elements)")

    def size(self) -> int:
        """|S₁ × … × S_k| = ∏ |S_i| (rule of product)."""
        return math.prod(len(p.values) for p in self.parameters)

    def formula(self) -> str:
        sizes = [str(len(p.values)) for p in self.parameters]
        return " × ".join(sizes) + f" = {self.size():,}"

    def enumerate(self) -> Iterator[dict[str, Any]]:
        keys = [p.key for p in self.parameters]
        for combo in itertools.product(*(p.values for p in self.parameters)):
            yield dict(zip(keys, combo))

    def is_valid(self, config: dict[str, Any]) -> bool:
        return all(c.check(config) for c in self.constraints)

    def valid_configs(self) -> list[dict[str, Any]]:
        return [c for c in self.enumerate() if self.is_valid(c)]

    def violated(self, config: dict[str, Any]) -> list[str]:
        return [c.description for c in self.constraints if not c.check(config)]

    def projection_size(self, groups: str | Sequence[str], configs: Sequence[dict[str, Any]] | None = None) -> int:
        """Number of distinct value-combinations of the parameters in the given group(s)."""
        groups = {groups} if isinstance(groups, str) else set(groups)
        keys = [p.key for p in self.parameters if p.group in groups]
        configs = configs if configs is not None else list(self.enumerate())
        return len({tuple(c[k] for k in keys) for c in configs})

    def label(self, config: dict[str, Any]) -> str:
        return ", ".join(f"{p.label}={p.fmt(config[p.key])}" for p in self.parameters)

    def short_label(self, config: dict[str, Any]) -> str:
        return "·".join(p.fmt(config[p.key]) for p in self.parameters)


def n_choose_k(n: int, k: int) -> int:
    return math.comb(n, k) if 0 <= k <= n else 0


def select_configs(configs: list[dict[str, Any]], limit: int, strategy: str = "first",
                   seed: int = 0) -> list[dict[str, Any]]:
    """Pick at most ``limit`` configurations: the first ones or a seeded random sample."""
    if limit < 1:
        raise ValueError("limit must be at least 1")
    if len(configs) <= limit:
        return list(configs)
    if strategy == "first":
        return list(configs[:limit])
    if strategy == "random":
        rng = random.Random(seed)
        idx = sorted(rng.sample(range(len(configs)), limit))
        return [configs[i] for i in idx]
    raise ValueError("strategy must be 'first' or 'random'")
