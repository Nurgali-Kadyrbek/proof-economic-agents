#!/usr/bin/env python3
"""Generate paired descriptive summaries from the frozen review rows."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def summarize(data):
    summaries = {}
    rng = np.random.default_rng(data['config']['bootstrap']['seed'])
    reps = data['config']['bootstrap']['repetitions']
    for cohort in data['config']['cohorts']:
        summaries[cohort] = {}
        for mode in data['config']['workloads']:
            rows = [r for r in data['theories'] if r['cohort'] == cohort and r['mode'] == mode]
            samples = rng.integers(0, len(rows), (reps, len(rows)))
            group = {}
            for control in ['lazy_first_use', 'one_step_certification_voi', 'workload_weighted_drd_hec', 'committed_bundle', 'exact_density']:
                reduction = np.asarray([1 - r['policies']['lpba']['completion_cost'] / r['policies'][control]['completion_cost'] for r in rows])
                area = np.asarray([r['policies']['lpba']['auc'] - r['policies'][control]['auc'] for r in rows])
                group[control] = {'lpba_completion_reduction': {'mean': float(reduction.mean()), 'ci95': list(np.quantile(reduction[samples].mean(axis=1), [.025, .975]))},
                                  'lpba_minus_control_area': {'mean': float(area.mean()), 'ci95': list(np.quantile(area[samples].mean(axis=1), [.025, .975]))},
                                  'lpba_wins_ties_losses': [sum(area > 1e-9), sum(abs(area) <= 1e-9), sum(area < -1e-9)]}
                group[control]['lpba_wins_ties_losses'] = [int(x) for x in group[control]['lpba_wins_ties_losses']]
            summaries[cohort][mode] = group
    return {'schema_version': 'acquisition-review-paired-summary-v1',
            'interpretation': 'secondary descriptive matched-theory comparisons; same cohorts and outcomes, no independent replication or multiplicity-adjusted confirmatory test',
            'summaries': summaries}


def main():
    ap = argparse.ArgumentParser();ap.add_argument('--output', type=Path, default=ROOT / 'results/acquisition_review/paired_summary.json');args = ap.parse_args()
    source = ROOT / 'results/acquisition_review/results.json'
    result = summarize(json.loads(source.read_text()))
    result['source_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
    result['generator_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    for cohort, modes in result['summaries'].items():
        print(cohort, {mode: group['exact_density'] for mode, group in modes.items()})


if __name__ == '__main__': main()
