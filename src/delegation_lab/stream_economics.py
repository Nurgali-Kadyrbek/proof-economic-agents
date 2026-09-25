"""Portfolio accounting for recurrent delegation experiments.

Operational task state may reset between episodes; evidence, certificates,
proof work, and maintenance are charged once to their persistent stream.
Task-level bootstraps must never resample those shared purchases.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .evaluation import paired_cluster_bootstrap


@dataclass(frozen=True, slots=True)
class PortfolioCost:
    evidence: float = 0.0
    proof: float = 0.0
    maintenance: float = 0.0

    @property
    def total(self) -> float:
        return self.evidence + self.proof + self.maintenance


def summarize_stream(runs: Sequence[Any], portfolio_cost: PortfolioCost) -> dict[str, float]:
    """Totals over one persistent arm portfolio, never replicated per task."""
    if not runs:
        raise ValueError("stream must have at least one episode")
    episodes = len(runs)
    total_calls = sum(row.agent_model_calls for row in runs)
    total_tokens = sum(row.agent_input_tokens + row.agent_output_tokens for row in runs)
    total_useful = sum(row.delegated_useful_workload for row in runs)
    source_obligations = sum(len(row.repair_obligations) for row in runs)
    online_certifiable = sum(len(set(row.repair_obligations).intersection(row.active_programs_after_episode)) for row in runs)
    return {
        "episodes": episodes,
        "eligible_episodes": sum(bool(row.eligible_recurrent_workload) for row in runs),
        "benchmark_successes": sum(bool(row.benchmark_success) for row in runs if row.benchmark_success is not None),
        "invalid_actions": sum(row.invalid_tool_actions for row in runs),
        "protocol_violations": sum(row.protocol_violations for row in runs),
        "model_calls": total_calls,
        "model_tokens": total_tokens,
        "deterministic_actions": sum(row.deterministic_actions for row in runs),
        "delegated_useful_workload": total_useful,
        "source_repair_obligations": source_obligations,
        "online_certifiable_repair_obligations": online_certifiable,
        "portfolio_evidence_cost": portfolio_cost.evidence,
        "portfolio_proof_cost": portfolio_cost.proof,
        "portfolio_maintenance_cost": portfolio_cost.maintenance,
        "portfolio_total_cost": portfolio_cost.total,
        "amortized_total_cost_per_episode_descriptive": portfolio_cost.total / episodes,
    }


def paired_stream_totals(left: Mapping[str, float], right: Mapping[str, float]) -> dict[str, float]:
    if left["episodes"] != right["episodes"]:
        raise ValueError("paired arms need the same number of episodes")
    return {
        "model_calls_displaced": right["model_calls"] - left["model_calls"],
        "tokens_displaced": right["model_tokens"] - left["model_tokens"],
        "success_difference": left["benchmark_successes"] - right["benchmark_successes"],
        "invalid_action_difference": left["invalid_actions"] - right["invalid_actions"],
        "protocol_violation_difference": left["protocol_violations"] - right["protocol_violations"],
        "evidence_cost_difference": left["portfolio_evidence_cost"] - right["portfolio_evidence_cost"],
        "proof_cost_difference": left["portfolio_proof_cost"] - right["portfolio_proof_cost"],
        "maintenance_cost_difference": left["portfolio_maintenance_cost"] - right["portfolio_maintenance_cost"],
    }


def independent_stream_cost_interval(
    left: Mapping[str, PortfolioCost], right: Mapping[str, PortfolioCost], *, seed: int = 19
) -> dict[str, float | int | str | None]:
    """Only independent streams are sampling units for acquisition economics."""
    common = set(left).intersection(right)
    if len(common) < 2:
        return {"ci95_low": None, "ci95_high": None, "streams": len(common), "reason": "at least two independent paired streams are required"}
    interval = paired_cluster_bootstrap(
        {key: left[key].total for key in common},
        {key: right[key].total for key in common},
        seed=seed,
    )
    return {**interval, "streams": len(common)}


def break_even_sensitivity(
    fixed_cost: PortfolioCost,
    *,
    calls_saved_per_episode: float,
    tokens_saved_per_episode: float,
    value_per_model_call: Sequence[float],
    value_per_thousand_tokens: Sequence[float],
    incremental_runtime_cost_per_episode: float = 0.0,
) -> list[dict[str, float | int | None]]:
    """Break-even episode counts over explicit, dimensionless cost ratios.

    Values are evidence/proof-cost units, not assumed dollars. ``None`` means
    no finite break-even under that conversion, including negative savings.
    """
    if incremental_runtime_cost_per_episode < 0 or any(x < 0 for x in (*value_per_model_call, *value_per_thousand_tokens)):
        raise ValueError("cost-conversion assumptions must be nonnegative")
    rows: list[dict[str, float | int | None]] = []
    for call_value in value_per_model_call:
        for token_value in value_per_thousand_tokens:
            recurring_value = calls_saved_per_episode * call_value + tokens_saved_per_episode / 1000 * token_value - incremental_runtime_cost_per_episode
            rows.append({
                "value_per_model_call": call_value,
                "value_per_thousand_tokens": token_value,
                "incremental_runtime_cost_per_episode": incremental_runtime_cost_per_episode,
                "net_savings_per_episode": recurring_value,
                "break_even_episodes": ceil(fixed_cost.total / recurring_value) if recurring_value > 0 else None,
            })
    return rows


def plot_ca_cost(curves: Mapping[str, Sequence[tuple[float, float]]], path: Path, *, ylabel: str = "Certified useful recurrent workload CA(K)") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 4.5), layout="constrained")
    for name, points in curves.items():
        if points:
            axis.step([point[0] for point in points], [point[1] for point in points], where="post", marker="o", label=name)
    axis.set(xlabel="Cumulative portfolio evidence cost (review units)", ylabel=ylabel)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_break_even(rows: Sequence[Mapping[str, float | int | None]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 4.5), layout="constrained")
    token_values = sorted({float(row["value_per_thousand_tokens"]) for row in rows})
    for token_value in token_values:
        selected = sorted((row for row in rows if row["value_per_thousand_tokens"] == token_value), key=lambda row: float(row["value_per_model_call"]))
        axis.plot(
            [float(row["value_per_model_call"]) for row in selected],
            [row["break_even_episodes"] if row["break_even_episodes"] is not None else float("nan") for row in selected],
            marker="o", label=f"1k tokens = {token_value:g} units",
        )
    axis.set(xlabel="Value per displaced model call (review-cost units)", ylabel="Additional episodes to break even")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.savefig(path, dpi=180)
    plt.close(figure)
