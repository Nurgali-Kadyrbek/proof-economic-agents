"""Machine-readable aggregation, paired cluster bootstrap, and compact plots."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from random import Random
from typing import Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .acquire import AcquisitionRun, AcquisitionStep


@dataclass(frozen=True, slots=True)
class CurvePoint:
    budget_fraction: float
    mean_cost: float
    mean_certified_weight: float
    mean_coverage: float


def sample_run(run: AcquisitionRun, fraction: float) -> AcquisitionStep | None:
    allowed = run.budget * fraction
    choices = [step for step in run.steps if step.cumulative_evidence_cost <= allowed + 1e-12]
    return choices[-1] if choices else None


def aggregate_curves(runs: Sequence[AcquisitionRun], fractions: Sequence[float]) -> list[CurvePoint]:
    result: list[CurvePoint] = []
    for fraction in fractions:
        costs: list[float] = []
        weights: list[float] = []
        coverage: list[float] = []
        for run in runs:
            step = sample_run(run, fraction)
            total = run.final.total_weight if run.final else 0.0
            costs.append(step.cumulative_evidence_cost if step else 0.0)
            weights.append(step.certified_weight if step else 0.0)
            coverage.append((step.certified_weight / total) if step and total else 0.0)
        result.append(CurvePoint(fraction, float(np.mean(costs)), float(np.mean(weights)), float(np.mean(coverage))))
    return result


def paired_cluster_bootstrap(
    left: Mapping[str, float], right: Mapping[str, float], *, replicates: int = 2000, seed: int = 19
) -> dict[str, float]:
    """Cluster bootstrap for matched original theories/task families."""
    common = sorted(set(left).intersection(right))
    if not common:
        raise ValueError("paired bootstrap needs common cluster ids")
    diffs = np.array([left[key] - right[key] for key in common], dtype=float)
    rng = Random(seed)
    samples = []
    for _ in range(replicates):
        indices = [rng.randrange(len(diffs)) for _ in diffs]
        samples.append(float(np.mean(diffs[indices])))
    return {
        "mean_difference": float(np.mean(diffs)),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "clusters": len(common),
        "replicates": replicates,
    }


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def write_summary_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_coverage_curves(curves: Mapping[str, Sequence[CurvePoint]], path: Path, *, title: str, x_label: str = "Evidence-cost budget fraction") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 4.8), layout="constrained")
    for name, curve in curves.items():
        axis.plot([point.budget_fraction for point in curve], [point.mean_coverage for point in curve], marker="o", label=name)
    axis.set(title=title, xlabel=x_label, ylabel="Useful certified workload / workload")
    axis.set_ylim(-0.02, 1.02)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, ncol=2)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _json_default(value: object) -> object:
    if hasattr(value, "value"):
        return getattr(value, "value")
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    if isinstance(value, set | frozenset):
        return sorted(value)
    raise TypeError(f"not JSON serializable: {type(value)!r}")
