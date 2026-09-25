"""Outcome-blind structural stress study on unused ProofWriter test theories."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from ..adapters.proofwriter import (
    PROOFWRITER_ARCHIVE_SHA256,
    load_proofwriter_theories,
    reweight_theories_by_predicate_prior,
)
from .proofwriter_exact_characterization import POLICIES, _summary
from .proofwriter_exact_core import Catalog, SupportLimitError, exact_oracle, feasible_auc, prefix_value


ROOT = Path(__file__).resolve().parents[3]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_freeze(config_path: Path) -> str:
    freeze_path = ROOT / "results/proofwriter_structural_stress/freeze_manifest.json"
    freeze = json.loads(freeze_path.read_text())
    expected = {
        "config": config_path,
        "protocol": ROOT / "docs/proofwriter_structural_stress_protocol.md",
        "core": ROOT / "src/delegation_lab/experiments/proofwriter_exact_core.py",
        "summary_runner": ROOT / "src/delegation_lab/experiments/proofwriter_exact_characterization.py",
        "runner": Path(__file__),
        "source_test": ROOT / "data/proofwriter/proofwriter-dataset-V2020.12.3/OWA/depth-5/meta-test.jsonl",
        "train_prior_artifact": ROOT / "results/proofwriter_train_prior/results.json",
    }
    if set(freeze["hashes"]) != set(expected):
        raise ValueError("structural-stress freeze has an unexpected file set")
    for label, path in expected.items():
        if freeze["hashes"][label] != _sha256(path):
            raise RuntimeError(f"structural-stress freeze mismatch: {label}")
    return _sha256(freeze_path)


def run(config_path: Path) -> dict[str, Any]:
    freeze_sha256 = _verify_freeze(config_path)
    config = yaml.safe_load(config_path.read_text())
    if not isinstance(config, Mapping):
        raise ValueError("structural-stress config must be a mapping")
    output = ROOT / config["output"]
    selection_output = output.with_name("selection_manifest.json")
    if output.exists() or selection_output.exists():
        raise FileExistsError("refusing to repeat structural-stress selection or evaluation")
    dataset = config["dataset"]
    selection = config["selection"]
    if dataset["archive_sha256"] != PROOFWRITER_ARCHIVE_SHA256:
        raise ValueError("config archive hash does not match the pinned ProofWriter release")
    if list(config["workloads"]) != ["uniform", "train_predicate_frequency"]:
        raise ValueError("workload list changed from the predeclared protocol")
    start = int(dataset["start_index"])
    end = int(dataset["selection_pool_end"])
    target_count = int(dataset["target_theories"])
    max_relevant = int(selection["max_relevant_evidence"])
    support_limit = int(selection["max_minimal_supports_per_literal"])
    if (start, end, target_count, max_relevant, support_limit) != (200, 947, 100, 18, 10000):
        raise ValueError("structural selection settings changed from the frozen protocol")
    if selection["rank"] != ["descending_relevant_evidence", "descending_minimal_support_count", "ascending_theory_id"]:
        raise ValueError("structural ranking changed from the frozen protocol")
    if selection["require_positive_complementarity"] is not True or selection["require_positive_shared_evidence"] is not True:
        raise ValueError("structural eligibility changed from the frozen protocol")
    source_root = ROOT / dataset["root"]
    theories = load_proofwriter_theories(
        source_root, split=str(dataset["split"]), depth=str(dataset["depth"]), limit=end
    )
    if len(theories) != end or len({theory.theory_id for theory in theories}) != end:
        raise ValueError("selection pool source order or identities are invalid")
    already_used_ids = {
        row["theory_id"]
        for row in json.loads((ROOT / "results/proofwriter_exact_characterization/results.json").read_text())["studies"]["uniform"]["theories"]
    }
    already_used_ids.update(
        row["theory_id"]
        for row in json.loads((ROOT / "results/proofwriter/results.json").read_text())["runs"]["lpba"]
    )
    if already_used_ids.intersection(theory.theory_id for theory in theories[start:end]):
        raise ValueError("structural pool overlaps an earlier analyzed cohort")
    eligible: list[tuple[int, int, str, int, Catalog]] = []
    support_limit_exclusions: list[dict[str, str]] = []
    pool_counts = {"total": end - start, "within_oracle_limit": 0, "positive_complementarity": 0, "positive_shared_evidence": 0, "eligible": 0}
    for source_index in range(start, end):
        theory = theories[source_index]
        try:
            catalog = Catalog.build(theory.candidates, theory.workload, support_limit=support_limit)
        except SupportLimitError as exc:
            support_limit_exclusions.append({"theory_id": theory.theory_id, "reason": str(exc)})
            continue
        features = catalog.feature_values()
        within_limit = catalog.relevant_count <= max_relevant
        complementarity = features["complementarity_fraction"] > 0
        shared = features["shared_evidence_cost_fraction"] > 0
        pool_counts["within_oracle_limit"] += within_limit
        pool_counts["positive_complementarity"] += complementarity
        pool_counts["positive_shared_evidence"] += shared
        if within_limit and complementarity and shared:
            pool_counts["eligible"] += 1
            eligible.append((-catalog.relevant_count, -catalog.support_count(), theory.theory_id, source_index, catalog))
        if (source_index - start + 1) % 100 == 0:
            print(f"selected pool: {source_index - start + 1}/{end-start}; eligible {len(eligible)}", flush=True)
    chosen = sorted(eligible)[:target_count]
    selected_ids = [entry[2] for entry in chosen]
    selection_manifest = {
        "selection_before_policy_evaluation": True,
        "selection_basis": "formal-source minimal-support structure and question predicates only",
        "pool_source_order": [start, end],
        "pool_counts": pool_counts,
        "support_limit_exclusions": support_limit_exclusions,
        "selected_theory_ids": selected_ids,
        "selected_theory_ids_sha256": hashlib.sha256(json.dumps(selected_ids).encode()).hexdigest(),
        "selected_source_indices": [entry[3] for entry in chosen],
        "freeze_manifest_sha256": freeze_sha256,
    }
    selection_output.parent.mkdir(parents=True, exist_ok=True)
    selection_output.write_text(json.dumps(selection_manifest, indent=2, sort_keys=True) + "\n")
    print(f"selection frozen: {len(chosen)} theories from {len(eligible)} eligible", flush=True)

    prior_artifact = json.loads((ROOT / "results/proofwriter_train_prior/results.json").read_text())
    prior_metadata = prior_artifact["workload"]
    if prior_metadata["prior_split"] != "train" or prior_metadata["labels_visible_to_prior"] is not False:
        raise ValueError("saved predicate prior has an unexpected provenance")
    prior = {str(key): float(value) for key, value in prior_metadata["predicate_prior"].items()}
    selected_theories = [theories[entry[3]] for entry in chosen]
    weighted_theories = reweight_theories_by_predicate_prior(selected_theories, prior)
    fractions = tuple(float(value) for value in config["oracle"]["budget_fractions"])
    rows: dict[str, list[dict[str, Any]]] = {"uniform": [], "train_predicate_frequency": []}
    for position, (entry, theory, weighted) in enumerate(zip(chosen, selected_theories, weighted_theories, strict=True), start=1):
        original = entry[4]
        for mode, workload in (("uniform", theory.workload), ("train_predicate_frequency", weighted.workload)):
            catalog = Catalog(original.ids, original.costs, tuple(workload), original.supports)
            total_cost = sum(catalog.costs)
            policy_rows: dict[str, Any] = {}
            for policy, chooser in POLICIES.items():
                order = chooser(catalog)
                policy_rows[policy] = {
                    "auc": feasible_auc(catalog, order),
                    "budget_value": {str(fraction): prefix_value(catalog, order, total_cost * fraction) for fraction in fractions},
                }
            policy_rows["eager_full_formalization"] = {
                "auc": 0.0,
                "budget_value": {str(fraction): 0.0 for fraction in fractions},
            }
            oracle = exact_oracle(catalog, fractions, max_relevant=max_relevant)
            if abs(oracle.auc - feasible_auc(catalog, oracle.order)) > 1e-8:
                raise RuntimeError("exact DP objective disagrees with reconstructed order")
            if any(policy_rows[policy]["auc"] > oracle.auc + 1e-8 for policy in POLICIES):
                raise RuntimeError("a policy exceeded the exact acquisition-order oracle")
            rows[mode].append({
                "theory_id": theory.theory_id,
                "source_order_index": entry[3],
                "candidate_count": len(catalog.ids),
                "minimal_support_count": catalog.support_count(),
                "relevant_evidence_count": catalog.relevant_count,
                "features": catalog.feature_values(),
                "exact": {"auc": oracle.auc, "budget_frontier": {str(fraction): oracle.budget_frontier[fraction] for fraction in fractions}},
                "policies": policy_rows,
            })
        if position % 10 == 0:
            print(f"evaluated: {position}/{len(chosen)} exact theories", flush=True)
    repetitions = int(config["bootstrap"]["repetitions"])
    seed = int(config["bootstrap"]["seed"])
    report = {
        "schema_version": "proofwriter-structural-stress-v1",
        "status": "outcome-blind formal-source structural stress cohort from the same ProofWriter test release; not average benchmark performance",
        "protocol": "docs/proofwriter_structural_stress_protocol.md",
        "freeze_manifest_sha256": freeze_sha256,
        "selection_manifest_sha256": _sha256(selection_output),
        "config_sha256": _sha256(config_path),
        "code_sha256": {"core": _sha256(ROOT / "src/delegation_lab/experiments/proofwriter_exact_core.py"), "runner": _sha256(Path(__file__))},
        "dataset": {
            "release": "V2020.12.3",
            "archive_sha256": PROOFWRITER_ARCHIVE_SHA256,
            "source_file_sha256": _sha256(source_root / "proofwriter-dataset-V2020.12.3" / "OWA" / str(dataset["depth"]) / "meta-test.jsonl"),
            "split": dataset["split"], "depth": dataset["depth"],
            "pool_source_order": [start, end], "selected_theories": len(chosen),
            "prior_source": "frozen results/proofwriter_train_prior/results.json; originally train questions, predicate frequencies, no labels",
            "gold_answers_or_proof_dags_visible_to_policies": False,
            "formal_source_oracle": True,
        },
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "oracle": {"max_relevant_evidence": max_relevant, "max_minimal_supports_per_literal": support_limit, "objective": "exact budget-feasible acquisition-order AUC on inclusion-minimal positive Horn supports"},
        "pool_counts": pool_counts,
        "support_limit_exclusions": support_limit_exclusions,
        "studies": {
            mode: {"summary": _summary(mode_rows, seed=seed + offset * 1000, repetitions=repetitions, fractions=fractions), "theories": mode_rows}
            for offset, (mode, mode_rows) in enumerate(rows.items())
        },
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/proofwriter_structural_stress.yaml"))
    args = parser.parse_args()
    result = run(args.config)
    for mode, study in result["studies"].items():
        gap = study["summary"]["relative_lpba_gap_to_exact"]
        print(mode, "theories", study["summary"]["theory_count"], "exact", study["summary"]["exact_theory_count"], "mean relative gap", gap["mean"])


if __name__ == "__main__":
    main()
