"""Predeclared fresh-cohort exact acquisition characterization on ProofWriter."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
from pathlib import Path
from statistics import mean
from typing import Any, Mapping

import numpy as np
import yaml

from ..adapters.proofwriter import (
    PROOFWRITER_ARCHIVE_SHA256,
    load_proofwriter_theories,
    reweight_theories_by_predicate_prior,
)
from .proofwriter_exact_core import (
    Catalog,
    SupportLimitError,
    decision_directed_order,
    exact_oracle,
    feasible_auc,
    lazy_first_use_order,
    lpba_order,
    one_step_order,
    prefix_value,
)


ROOT = Path(__file__).resolve().parents[3]
POLICIES = {
    "lpba": lpba_order,
    "lazy_first_use": lazy_first_use_order,
    "one_step_certification_voi": one_step_order,
    "workload_weighted_drd_hec": decision_directed_order,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_freeze(config_path: Path) -> str:
    freeze_path = ROOT / "results/proofwriter_exact_characterization/freeze_manifest.json"
    freeze = json.loads(freeze_path.read_text())
    expected = {
        "config": config_path,
        "protocol": ROOT / "docs/proofwriter_exact_characterization_protocol.md",
        "core": ROOT / "src/delegation_lab/experiments/proofwriter_exact_core.py",
        "runner": Path(__file__),
        "source_test": ROOT / "data/proofwriter/proofwriter-dataset-V2020.12.3/OWA/depth-5/meta-test.jsonl",
        "train_prior_artifact": ROOT / "results/proofwriter_train_prior/results.json",
    }
    if set(freeze["hashes"]) != set(expected):
        raise ValueError("fresh-cohort freeze has an unexpected file set")
    for label, path in expected.items():
        if freeze["hashes"][label] != _sha256(path):
            raise RuntimeError(f"fresh-cohort freeze mismatch: {label}")
    return _sha256(freeze_path)


def _interval(values: list[float], *, seed: int, repetitions: int) -> dict[str, Any]:
    if not values:
        return {"mean": None, "ci95": [None, None], "theory_count": 0}
    rng = random.Random(seed)
    draws = sorted(
        mean(values[rng.randrange(len(values))] for _ in values)
        for _ in range(repetitions)
    )
    return {
        "mean": mean(values),
        "ci95": [draws[int(0.025 * repetitions)], draws[min(repetitions - 1, int(0.975 * repetitions))]],
        "theory_count": len(values),
        "bootstrap_repetitions": repetitions,
        "bootstrap_unit": "theory",
    }


def _ranks(values: list[float]) -> list[float]:
    positions = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[positions[end]] == values[positions[start]]:
            end += 1
        rank = (start + end - 1) / 2 + 1
        for offset in range(start, end):
            ranks[positions[offset]] = rank
        start = end
    return ranks


def _spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) < 3 or len(left) != len(right):
        return None
    x, y = np.asarray(_ranks(left)), np.asarray(_ranks(right))
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _summary(rows: list[dict[str, Any]], *, seed: int, repetitions: int, fractions: tuple[float, ...]) -> dict[str, Any]:
    exact_rows = [row for row in rows if row["exact"] is not None]
    positive_rows = [row for row in exact_rows if row["exact"]["auc"] > 0]
    gaps = [
        (row["exact"]["auc"] - row["policies"]["lpba"]["auc"]) / row["exact"]["auc"]
        for row in positive_rows
    ]
    paired = {
        control: _interval(
            [row["policies"]["lpba"]["auc"] - row["policies"][control]["auc"] for row in rows],
            seed=seed + index,
            repetitions=repetitions,
        )
        for index, control in enumerate(("lazy_first_use", "one_step_certification_voi", "workload_weighted_drd_hec", "eager_full_formalization"))
    }
    associations = {
        feature: _spearman([row["features"][feature] for row in positive_rows], gaps)
        for feature in ("complementarity_fraction", "shared_evidence_cost_fraction", "workload_weight_cv")
    }
    budgets = {
        str(fraction): {
            "exact_mean_weight": mean(row["exact"]["budget_frontier"][str(fraction)] for row in exact_rows) if exact_rows else None,
            "policy_mean_weight": {
                policy: mean(row["policies"][policy]["budget_value"][str(fraction)] for row in exact_rows) if exact_rows else None
                for policy in (*POLICIES, "eager_full_formalization")
            },
        }
        for fraction in fractions
    }
    return {
        "theory_count": len(rows),
        "exact_theory_count": len(exact_rows),
        "positive_optimum_theory_count": len(positive_rows),
        "mean_auc": {
            "exact": mean(row["exact"]["auc"] for row in exact_rows) if exact_rows else None,
            **{policy: mean(row["policies"][policy]["auc"] for row in rows) for policy in (*POLICIES, "eager_full_formalization")},
        },
        "relative_lpba_gap_to_exact": _interval(gaps, seed=seed + 100, repetitions=repetitions),
        "worst_relative_lpba_gap": max(gaps) if gaps else None,
        "exact_lpba_match_count": sum(abs(row["exact"]["auc"] - row["policies"]["lpba"]["auc"]) <= 1e-9 for row in exact_rows),
        "lpba_minus_control_auc": paired,
        "spearman_relative_gap_associations": associations,
        "budget_frontiers": budgets,
    }


def run(config_path: Path) -> dict[str, Any]:
    freeze_sha256 = _verify_freeze(config_path)
    config = yaml.safe_load(config_path.read_text())
    if not isinstance(config, Mapping):
        raise ValueError("characterization config must be a mapping")
    output = ROOT / config["output"]
    if output.exists():
        raise FileExistsError(f"refusing to repeat fresh cohort: {output}")
    dataset = config["dataset"]
    if dataset["archive_sha256"] != PROOFWRITER_ARCHIVE_SHA256:
        raise ValueError("config archive hash does not match the pinned ProofWriter release")
    if list(config["comparators"]) != [*POLICIES, "eager_full_formalization"]:
        raise ValueError("comparator list changed from the predeclared protocol")
    if list(config["workloads"]) != ["uniform", "train_predicate_frequency"]:
        raise ValueError("workload list changed from the predeclared protocol")
    source_root = ROOT / dataset["root"]
    start = int(dataset["start_index"])
    count = int(dataset["theory_count"])
    if start != 100 or count != 100:
        raise ValueError("fresh cohort slice changed from the predeclared protocol")
    theories = load_proofwriter_theories(
        source_root,
        split=str(dataset["split"]),
        depth=str(dataset["depth"]),
        limit=start + count,
    )[start : start + count]
    if len(theories) != count or len({theory.theory_id for theory in theories}) != count:
        raise ValueError("fresh cohort size or identity is invalid")
    old_ids = {
        run["theory_id"]
        for run in json.loads((ROOT / "results/proofwriter/results.json").read_text())["runs"]["lpba"]
    }
    if old_ids.intersection(theory.theory_id for theory in theories):
        raise ValueError("fresh cohort overlaps the earlier 100-theory analysis")
    prior_artifact = json.loads((ROOT / "results/proofwriter_train_prior/results.json").read_text())
    prior_metadata = prior_artifact["workload"]
    if prior_metadata["prior_split"] != "train" or prior_metadata["labels_visible_to_prior"] is not False:
        raise ValueError("saved predicate prior has an unexpected provenance")
    prior = {str(key): float(value) for key, value in prior_metadata["predicate_prior"].items()}
    weighted_theories = reweight_theories_by_predicate_prior(theories, prior)
    fraction_values = tuple(float(value) for value in config["oracle"]["budget_fractions"])
    max_relevant = int(config["oracle"]["max_relevant_evidence"])
    support_limit = int(config["oracle"]["max_minimal_supports_per_literal"])
    rows: dict[str, list[dict[str, Any]]] = {"uniform": [], "train_predicate_frequency": []}
    exclusions: list[dict[str, str]] = []
    for position, (theory, weighted) in enumerate(zip(theories, weighted_theories, strict=True), start=1):
        try:
            original = Catalog.build(theory.candidates, theory.workload, support_limit=support_limit)
        except SupportLimitError as exc:
            exclusions.append({"theory_id": theory.theory_id, "reason": str(exc)})
            print(f"{position}/{count}: support limit; {len(exclusions)} excluded", flush=True)
            continue
        for mode, workload in (("uniform", theory.workload), ("train_predicate_frequency", weighted.workload)):
            catalog = Catalog(original.ids, original.costs, tuple(workload), original.supports)
            total_cost = sum(catalog.costs)
            policy_rows: dict[str, Any] = {}
            for policy, chooser in POLICIES.items():
                order = chooser(catalog)
                policy_rows[policy] = {
                    "auc": feasible_auc(catalog, order),
                    "budget_value": {str(fraction): prefix_value(catalog, order, total_cost * fraction) for fraction in fraction_values},
                }
            policy_rows["eager_full_formalization"] = {
                "auc": 0.0,
                "budget_value": {str(fraction): 0.0 for fraction in fraction_values},
            }
            try:
                oracle = exact_oracle(catalog, fraction_values, max_relevant=max_relevant)
            except SupportLimitError:
                exact = None
            else:
                if abs(oracle.auc - feasible_auc(catalog, oracle.order)) > 1e-8:
                    raise RuntimeError("exact DP objective disagrees with reconstructed order")
                if any(policy_rows[policy]["auc"] > oracle.auc + 1e-8 for policy in POLICIES):
                    raise RuntimeError("a policy exceeded the exact acquisition-order oracle")
                exact = {
                    "auc": oracle.auc,
                    "budget_frontier": {str(fraction): oracle.budget_frontier[fraction] for fraction in fraction_values},
                }
            rows[mode].append({
                "theory_id": theory.theory_id,
                "source_order_index": start + position - 1,
                "candidate_count": len(catalog.ids),
                "minimal_support_count": catalog.support_count(),
                "relevant_evidence_count": catalog.relevant_count,
                "features": catalog.feature_values(),
                "exact": exact,
                "policies": policy_rows,
            })
        if position % 10 == 0:
            print(f"{position}/{count}: exact {sum(row['exact'] is not None for row in rows['uniform'])}, support limits {len(exclusions)}", flush=True)
    repetitions = int(config["bootstrap"]["repetitions"])
    seed = int(config["bootstrap"]["seed"])
    report = {
        "schema_version": "proofwriter-exact-characterization-v1",
        "status": "fresh nonoverlapping source-order cohort from the same ProofWriter test release; not an independent benchmark",
        "protocol": "docs/proofwriter_exact_characterization_protocol.md",
        "freeze_manifest_sha256": freeze_sha256,
        "config_sha256": _sha256(config_path),
        "code_sha256": {
            "core": _sha256(ROOT / "src/delegation_lab/experiments/proofwriter_exact_core.py"),
            "runner": _sha256(Path(__file__)),
        },
        "dataset": {
            "release": "V2020.12.3",
            "archive_sha256": PROOFWRITER_ARCHIVE_SHA256,
            "source_file_sha256": _sha256(source_root / "proofwriter-dataset-V2020.12.3" / "OWA" / str(dataset["depth"]) / "meta-test.jsonl"),
            "split": dataset["split"],
            "depth": dataset["depth"],
            "start_index": start,
            "requested_theories": count,
            "prior_source": "frozen results/proofwriter_train_prior/results.json; originally train questions, predicate frequencies, no labels",
            "gold_answers_or_proof_dags_visible_to_policies": False,
            "formal_source_oracle": True,
        },
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "oracle": {
            "max_relevant_evidence": max_relevant,
            "max_minimal_supports_per_literal": support_limit,
            "objective": "exact budget-feasible acquisition-order AUC on inclusion-minimal positive Horn supports",
        },
        "exclusions": exclusions,
        "studies": {
            mode: {
                "summary": _summary(mode_rows, seed=seed + offset * 1000, repetitions=repetitions, fractions=fraction_values),
                "theories": mode_rows,
            }
            for offset, (mode, mode_rows) in enumerate(rows.items())
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/proofwriter_exact_characterization.yaml"))
    args = parser.parse_args()
    result = run(args.config)
    for mode, study in result["studies"].items():
        gap = study["summary"]["relative_lpba_gap_to_exact"]
        print(mode, "theories", study["summary"]["theory_count"], "exact", study["summary"]["exact_theory_count"], "mean relative gap", gap["mean"])


if __name__ == "__main__":
    main()
