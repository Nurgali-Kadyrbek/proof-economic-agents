#!/usr/bin/env python3
"""Rebuild all reported support catalogs and independently audit latency.

Use --data-root to locate a checked copy of the pinned ProofWriter release.
Frozen historical result files are never modified. Optional numba accelerates
the independent backward DP; it is not needed for artifact regeneration.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import random
from statistics import mean
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from delegation_lab.adapters.proofwriter import load_proofwriter_theories, reweight_theories_by_predicate_prior
from delegation_lab.experiments.proofwriter_exact_core import Catalog, feasible_auc
from delegation_lab.experiments.proofwriter_exact_characterization import POLICIES
from delegation_lab.experiments.acquisition_review import committed_bundle_order, exact_density_order, weighted_completion_cost


def independent_latency_dp(values, costs, count):
    # Backward completion-increment recurrence, independently derived from the
    # forward area DP used to produce the frozen optimum.
    dp = np.full(len(values), np.inf)
    dp[0] = 0.0
    for mask in range(1, len(values)):
        available = mask
        while available:
            bit = available & -available
            before = mask ^ bit
            candidate = dp[before] + costs[mask] * (values[mask] - values[before])
            if candidate < dp[mask]:
                dp[mask] = candidate
            available ^= bit
    return dp[-1]


try:
    from numba import njit
except ImportError:
    ACCELERATOR = 'python'
else:
    independent_latency_dp = njit(cache=True)(independent_latency_dp)
    ACCELERATOR = 'numba'


def exact_completion(catalog):
    indices = [i for i in range(len(catalog.ids)) if catalog.relevant_mask & (1 << i)]
    masks = np.arange(1 << len(indices), dtype=np.uint32)
    values = np.zeros(len(masks))
    costs = np.zeros(len(masks))
    for k, i in enumerate(indices):
        costs += catalog.costs[i] * ((masks >> k) & 1)
    for item in catalog.workload:
        covered = np.zeros(len(masks), dtype=bool)
        for support in catalog.supports[item.item_id]:
            converted = sum(1 << k for k, i in enumerate(indices) if support & (1 << i))
            covered |= (masks & converted) == converted
        values += item.weight * covered
    return float(independent_latency_dp(values, costs, len(indices)))


def evaluate(task):
    cohort, original, theory, weighted = task
    catalog = Catalog.build(theory.candidates, theory.workload, support_limit=10000)
    if catalog.relevant_count != original['relevant_evidence_count'] or catalog.support_count() != original['minimal_support_count']:
        raise ValueError('reconstructed support catalog does not match frozen artifact')
    rows = []
    for mode, workload in [('uniform', theory.workload), ('train_predicate_frequency', weighted.workload)]:
        cat = Catalog(catalog.ids, catalog.costs, tuple(workload), catalog.supports)
        frozen = FROZEN[cohort]['studies'][mode]['theories']
        old = next(x for x in frozen if x['theory_id'] == theory.theory_id)
        total_cost = sum(cat.costs)
        maximum = cat.value((1 << len(cat.ids)) - 1)
        exact_cost = exact_completion(cat)
        implied = total_cost * (maximum - old['exact']['auc'])
        if not np.isclose(exact_cost, implied, atol=1e-8, rtol=0):
            raise ValueError('independent backward latency DP disagrees with frozen forward area optimum')
        policies = {}
        for name, selector in {**POLICIES, 'committed_bundle': committed_bundle_order, 'exact_density': exact_density_order}.items():
            order = selector(cat)
            area = feasible_auc(cat, order)
            completion = weighted_completion_cost(cat, order)
            if not np.isclose(completion, total_cost * (maximum - area), atol=1e-8, rtol=0):
                raise ValueError('area and direct first-certification cost disagree')
            if name in POLICIES and not np.isclose(area, old['policies'][name]['auc'], atol=1e-8, rtol=0):
                raise ValueError('policy reproduction disagrees with frozen area')
            if completion < exact_cost - 1e-8:
                raise ValueError('policy exceeds independently verified optimum')
            policies[name] = {'auc': area, 'completion_cost': completion,
                              'relative_completion_excess': (completion - exact_cost) / exact_cost if exact_cost else None,
                              'relative_area_regret': (old['exact']['auc'] - area) / old['exact']['auc'],
                              'order': list(order)}
        rows.append({'cohort': cohort, 'mode': mode, 'theory_id': theory.theory_id,
                     'candidate_count': len(cat.ids), 'relevant_count': cat.relevant_count,
                     'total_cost': total_cost, 'relevant_cost': sum(c for i, c in enumerate(cat.costs) if cat.relevant_mask & (1 << i)),
                     'certifiable_weight': maximum, 'exact_completion_cost': exact_cost,
                     'independent_oracle_match': True, 'policies': policies})
    return rows


def interval(values, seed, repetitions):
    rng = np.random.default_rng(seed)
    data = np.asarray(values)
    samples = data[rng.integers(0, len(data), size=(repetitions, len(data)))].mean(axis=1)
    return {'mean': float(data.mean()), 'ci95': list(np.quantile(samples, [.025, .975])), 'n': len(data)}


FROZEN = {name: json.loads((ROOT / 'results' / name / 'results.json').read_text())
          for name in ['proofwriter_exact_characterization', 'proofwriter_structural_stress']}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', type=Path, default=ROOT / 'data/proofwriter')
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()
    config_path = ROOT / 'configs/acquisition_review.json'
    config = json.loads(config_path.read_text())
    output = ROOT / config['output']
    if output.exists():
        raise FileExistsError('refusing to overwrite the frozen review result')
    source = args.data_root / 'proofwriter-dataset-V2020.12.3/OWA/depth-5/meta-test.jsonl'
    if not source.exists():
        source = args.data_root / 'OWA/depth-5/meta-test.jsonl'
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    for old in FROZEN.values():
        if source_sha != old['dataset']['source_file_sha256']:
            raise ValueError('source checksum differs from the reported exact studies')
    theories = load_proofwriter_theories(args.data_root, split='test', depth='depth-5', limit=947)
    prior = json.loads((ROOT / 'results/proofwriter_train_prior/results.json').read_text())['workload']['predicate_prior']
    weighted = reweight_theories_by_predicate_prior(theories, prior)
    tasks = []
    for cohort in config['cohorts']:
        for old in FROZEN[cohort]['studies']['uniform']['theories']:
            index = old['source_order_index']
            if theories[index].theory_id != old['theory_id']:
                raise ValueError('source order or theory identity changed')
            tasks.append((cohort, old, theories[index], weighted[index]))
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for k, result in enumerate(pool.map(evaluate, tasks), 1):
            rows.extend(result)
            if k % 20 == 0:
                print(f'{k}/{len(tasks)} theories independently verified', flush=True)
    summaries = {}
    seed = config['bootstrap']['seed']; reps = config['bootstrap']['repetitions']
    for cohort in config['cohorts']:
        summaries[cohort] = {}
        for mode in config['workloads']:
            group = [x for x in rows if x['cohort'] == cohort and x['mode'] == mode]
            summary = {}
            for name in config['policies']:
                summary[name] = {
                    'relative_completion_excess': interval([x['policies'][name]['relative_completion_excess'] for x in group], seed, reps),
                    'relative_area_regret': interval([x['policies'][name]['relative_area_regret'] for x in group], seed, reps),
                    'completion_reduction_vs_lazy': interval([1 - x['policies'][name]['completion_cost'] / x['policies']['lazy_first_use']['completion_cost'] for x in group], seed, reps),
                    'mean_completion_cost': mean(x['policies'][name]['completion_cost'] for x in group),
                    'mean_auc': mean(x['policies'][name]['auc'] for x in group),
                    'exact_match_count': sum(abs(x['policies'][name]['completion_cost'] - x['exact_completion_cost']) < 1e-8 for x in group)}
            summaries[cohort][mode] = summary
    result = {'schema_version': 'acquisition-review-v1', 'config': config,
              'config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
              'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'implementation_sha256': hashlib.sha256((ROOT / 'src/delegation_lab/experiments/acquisition_review.py').read_bytes()).hexdigest(),
              'source_sha256': source_sha,
              'frozen_result_sha256': {name: hashlib.sha256((ROOT / 'results' / name / 'results.json').read_bytes()).hexdigest() for name in FROZEN},
              'acceleration': ACCELERATOR, 'summaries': summaries, 'theories': rows,
              'independent_oracle_matches': len(rows)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    for cohort, modes in summaries.items():
        for mode, s in modes.items():
            print(cohort, mode, {name: {'area_regret': round(v['relative_area_regret']['mean'] * 100, 3),
                                     'completion_excess': round(v['relative_completion_excess']['mean'] * 100, 3),
                                     'reduction_vs_lazy': round(v['completion_reduction_vs_lazy']['mean'] * 100, 3)} for name, v in s.items()}, flush=True)


if __name__ == '__main__':
    main()
