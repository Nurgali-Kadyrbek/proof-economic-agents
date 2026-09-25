"""Controlled demand-weight sensitivity on the frozen structural-stress cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

from ..adapters.proofwriter import load_proofwriter_theories
from .proofwriter_exact_characterization import POLICIES, _interval
from .proofwriter_exact_core import Catalog, exact_oracle, feasible_auc


ROOT = Path(__file__).resolve().parents[3]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_freeze(config_path: Path) -> str:
    freeze_path = ROOT / "results/proofwriter_workload_skew_sensitivity/freeze_manifest.json"
    freeze = json.loads(freeze_path.read_text())
    expected = {
        "config": config_path,
        "protocol": ROOT / "docs/proofwriter_workload_skew_sensitivity_protocol.md",
        "core": ROOT / "src/delegation_lab/experiments/proofwriter_exact_core.py",
        "summary_runner": ROOT / "src/delegation_lab/experiments/proofwriter_exact_characterization.py",
        "runner": Path(__file__),
        "source_study": ROOT / "results/proofwriter_structural_stress/results.json",
        "source_selection": ROOT / "results/proofwriter_structural_stress/selection_manifest.json",
        "source_test": ROOT / "data/proofwriter/proofwriter-dataset-V2020.12.3/OWA/depth-5/meta-test.jsonl",
    }
    if set(freeze["hashes"]) != set(expected):
        raise ValueError("workload-skew freeze has an unexpected file set")
    for label, path in expected.items():
        if freeze["hashes"][label] != _sha256(path):
            raise RuntimeError(f"workload-skew freeze mismatch: {label}")
    return _sha256(freeze_path)


def _weighted_catalog(base: Catalog, alpha: float, seed: int) -> Catalog:
    count = len(base.workload)
    ranked_ids = sorted(
        (item.item_id for item in base.workload),
        key=lambda item_id: hashlib.sha256(f"{seed}|{item_id}".encode()).digest(),
    )
    ranks = {item_id: rank for rank, item_id in enumerate(ranked_ids)}
    raw = [math.exp(alpha * (2 * ranks[item.item_id] / (count - 1) - 1)) if count > 1 else 1.0
           for item in base.workload]
    normalizer = mean(raw) if raw else 1.0
    workload = tuple(replace(item, weight=value / normalizer) for item, value in zip(base.workload, raw, strict=True))
    return Catalog(base.ids, base.costs, workload, base.supports)


def _evaluate_theory(task: tuple[Any, tuple[float, ...], tuple[int, ...], int]) -> dict[str, Any]:
    theory, alphas, seeds, support_limit = task
    base = Catalog.build(theory.candidates, theory.workload, support_limit=support_limit)
    scenarios = []
    for alpha in alphas:
        for seed in (seeds[:1] if alpha == 0 else seeds):
            catalog = _weighted_catalog(base, alpha, seed)
            oracle = exact_oracle(catalog, (), max_relevant=18)
            scores = {policy: feasible_auc(catalog, chooser(catalog)) for policy, chooser in POLICIES.items()}
            if any(score > oracle.auc + 1e-8 for score in scores.values()):
                raise RuntimeError("policy exceeded exact optimum")
            scenarios.append({
                "alpha": alpha, "seed": seed, "observed_weight_cv": catalog.feature_values()["workload_weight_cv"],
                "exact_auc": oracle.auc, "policy_auc": scores,
                "lpba_relative_gap": (oracle.auc - scores["lpba"]) / oracle.auc if oracle.auc > 0 else None,
            })
    return {"theory_id": theory.theory_id, "relevant_evidence_count": base.relevant_count, "scenarios": scenarios}


def run(config_path: Path) -> dict[str, Any]:
    freeze_sha256 = _verify_freeze(config_path)
    config = yaml.safe_load(config_path.read_text())
    output = ROOT / config["output"]
    if output.exists():
        raise FileExistsError(output)
    alphas = tuple(float(value) for value in config["weighting"]["alphas"])
    seeds = tuple(int(value) for value in config["weighting"]["hash_seeds"])
    if alphas != (0.0, 0.75, 1.5, 3.0) or seeds != (11, 23, 47):
        raise ValueError("weight conditions changed from protocol")
    if config["weighting"]["transform"] != "exp_of_centered_hash_rank_normalized_to_mean_one":
        raise ValueError("weight transform changed from protocol")
    if int(config["oracle"]["max_relevant_evidence"]) != 18 or int(config["oracle"]["max_minimal_supports_per_literal"]) != 10000:
        raise ValueError("oracle limits changed from protocol")
    if int(config["workers"]) != 16:
        raise ValueError("parallelism changed from frozen config")
    selection = json.loads((ROOT / config["source_selection"]).read_text())
    study = json.loads((ROOT / config["source_study"]).read_text())
    selected_ids = selection["selected_theory_ids"]
    if len(selected_ids) != 100 or set(selected_ids) != {row["theory_id"] for row in study["studies"]["uniform"]["theories"]}:
        raise ValueError("selected theories differ from the frozen stress study")
    source = load_proofwriter_theories(ROOT / config["dataset_root"], split="test", depth="depth-5", limit=947)
    by_id = {theory.theory_id: theory for theory in source}
    selected = [by_id[theory_id] for theory_id in selected_ids]
    tasks = [(theory, alphas, seeds, int(config["oracle"]["max_minimal_supports_per_literal"])) for theory in selected]
    by_result_id: dict[str, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=int(config["workers"])) as executor:
        futures = [executor.submit(_evaluate_theory, task) for task in tasks]
        for future in as_completed(futures):
            row = future.result()
            by_result_id[row["theory_id"]] = row
            if len(by_result_id) % 10 == 0:
                print(f"evaluated {len(by_result_id)}/{len(selected)} theories", flush=True)
    rows = [by_result_id[theory_id] for theory_id in selected_ids]
    repetitions = int(config["bootstrap"]["repetitions"])
    seed = int(config["bootstrap"]["seed"])
    summaries = {}
    for index, alpha in enumerate(alphas):
        per_theory = []
        for row in rows:
            scenarios = [scenario for scenario in row["scenarios"] if scenario["alpha"] == alpha]
            per_theory.append({
                "theory_id": row["theory_id"],
                "observed_weight_cv": mean(scenario["observed_weight_cv"] for scenario in scenarios),
                "lpba_relative_gap": mean(scenario["lpba_relative_gap"] for scenario in scenarios),
                "exact_auc": mean(scenario["exact_auc"] for scenario in scenarios),
                "policy_auc": {policy: mean(scenario["policy_auc"][policy] for scenario in scenarios) for policy in POLICIES},
            })
        summaries[str(alpha)] = {
            "theory_count": len(per_theory),
            "seed_count_per_theory": 1 if alpha == 0 else len(seeds),
            "mean_observed_weight_cv": mean(row["observed_weight_cv"] for row in per_theory),
            "lpba_relative_gap": _interval([row["lpba_relative_gap"] for row in per_theory], seed=seed + index * 10, repetitions=repetitions),
            "lpba_minus_lazy_auc": _interval([row["policy_auc"]["lpba"] - row["policy_auc"]["lazy_first_use"] for row in per_theory], seed=seed + index * 10 + 1, repetitions=repetitions),
            "lpba_minus_one_step_auc": _interval([row["policy_auc"]["lpba"] - row["policy_auc"]["one_step_certification_voi"] for row in per_theory], seed=seed + index * 10 + 2, repetitions=repetitions),
            "mean_exact_auc": mean(row["exact_auc"] for row in per_theory),
            "mean_policy_auc": {policy: mean(row["policy_auc"][policy] for row in per_theory) for policy in POLICIES},
        }
    report = {
        "schema_version": "proofwriter-workload-skew-sensitivity-v1",
        "status": "controlled post-hoc demand-weight sensitivity on the frozen structural-stress theories; not a fresh benchmark",
        "protocol": "docs/proofwriter_workload_skew_sensitivity_protocol.md",
        "freeze_manifest_sha256": freeze_sha256,
        "config_sha256": _sha256(config_path),
        "source_study_sha256": _sha256(ROOT / config["source_study"]),
        "source_selection_sha256": _sha256(ROOT / config["source_selection"]),
        "runner_sha256": _sha256(Path(__file__)),
        "environment": {"python": platform.python_version(), "workers": int(config["workers"])},
        "alphas": alphas, "hash_seeds": seeds,
        "summaries": summaries, "theories": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/proofwriter_workload_skew_sensitivity.yaml"))
    args = parser.parse_args()
    result = run(args.config)
    for alpha, summary in result["summaries"].items():
        print(alpha, summary["mean_observed_weight_cv"], summary["lpba_relative_gap"]["mean"])


if __name__ == "__main__":
    main()
