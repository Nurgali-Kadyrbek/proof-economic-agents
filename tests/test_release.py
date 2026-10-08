from __future__ import annotations

import json
import importlib.util
import itertools
import math
import os
import subprocess
import sys
import tempfile
import unittest
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from delegation_lab.acquire import WorkItem
from delegation_lab.ir import Literal
from delegation_lab.experiments.proofwriter_exact_core import Catalog, exact_oracle, feasible_auc, lpba_order
from delegation_lab.experiments.acquisition_review import weighted_completion_cost, exact_density_order, committed_bundle_order, brute_force_optimal_latency


class ReleaseChecks(unittest.TestCase):
    def test_generated_inputs_match_frozen_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run([sys.executable, str(ROOT / 'scripts/generate_manuscript_inputs.py'), '--output', tmp], check=True)
            expected = sorted((ROOT / 'generated').glob('*.tex'))
            self.assertEqual([p.name for p in expected], sorted(p.name for p in Path(tmp).glob('*.tex')))
            for p in expected:
                self.assertEqual(p.read_bytes(), (Path(tmp) / p.name).read_bytes(), p.name)

    def test_exact_oracle_and_lpba_on_complementary_support(self):
        # Source a alone gives one unit; a and b together grant three more.
        items = (WorkItem('solo', Literal('solo', ()), 'toy', 1.0),
                 WorkItem('joint', Literal('joint', ()), 'toy', 3.0))
        cat = Catalog(ids=('a', 'b'), costs=(1.0, 1.0), workload=items,
                      supports={'solo': (1,), 'joint': (3,)})
        exact = exact_oracle(cat, (0.5,), max_relevant=2)
        self.assertTrue(math.isclose(exact.auc, 0.5))
        self.assertEqual(exact.budget_frontier[0.5], 1.0)
        self.assertTrue(math.isclose(feasible_auc(cat, lpba_order(cat)), exact.auc))

    def test_banking_separable_oracles(self):
        d = json.loads((ROOT / 'results/banking_exact_separable/results.json').read_text())
        ids = d['ratio_order']; costs = d['packet_cost']; vals = d['packet_value']; core = d['common_core_cost']
        self.assertTrue(d['assumptions_verified']['noncore_family_supports_disjoint'])
        self.assertEqual(len(ids), d['assumptions_verified']['family_count'])
        self.assertTrue(math.isclose(core + sum(costs.values()), d['total_cost']))
        current = 0.0; area = 0.0
        for k in ids:
            area += current * costs[k]
            current += vals[k]
        self.assertTrue(math.isclose(area / d['total_cost'], d['ratio_order_auc']))
        for key, frontier in d['budget_frontiers'].items():
            budget = frontier['budget']
            optimum = 0.0
            for mask in range(1 << len(ids)):
                cost = core + sum(costs[ids[i]] for i in range(len(ids)) if mask & (1 << i))
                if cost <= budget + 1e-9:
                    value = sum(vals[ids[i]] for i in range(len(ids)) if mask & (1 << i))
                    optimum = max(optimum, value)
            self.assertEqual(optimum, frontier['exact_value'], key)

    def test_reported_cohorts_are_disjoint_and_exact(self):
        f = json.loads((ROOT / 'results/proofwriter_exact_characterization/results.json').read_text())
        s = json.loads((ROOT / 'results/proofwriter_structural_stress/results.json').read_text())
        prior = json.loads((ROOT / 'results/proofwriter/results.json').read_text())
        getids = lambda d: {x['theory_id'] for x in d['studies']['uniform']['theories']}
        a,b=getids(f),getids(s)
        c={x['theory_id'] for x in prior['runs']['lpba']}
        self.assertEqual(len(a),100);self.assertEqual(len(b),100);self.assertEqual(len(c),100)
        self.assertFalse(a & b or a & c or b & c)
        self.assertEqual(f['studies']['uniform']['summary']['exact_theory_count'],100)
        self.assertEqual(s['studies']['uniform']['summary']['exact_theory_count'],100)
        self.assertFalse(f['exclusions']);self.assertFalse(s['support_limit_exclusions'])

    def test_public_episode_trace_reconciles_with_aggregate(self):
        trace = json.loads((ROOT / 'results/tau2_telecom_stream_heldout/public_trace.json').read_text())
        summary = json.loads((ROOT / 'results/tau2_telecom_stream_heldout/public_summary.json').read_text())
        self.assertEqual(trace['schema_version'], 'public-episode-metrics-v1')
        for arm, rows in trace['stream_arms'].items():
            self.assertEqual(len(rows), 40)
            self.assertEqual([x['episode'] for x in rows], list(range(1, 41)))
            self.assertEqual(sum(x['agent_model_calls'] for x in rows), summary['stream']['stream_summary'][arm]['model_calls'])
        phases = trace['lifecycle_arms']['qwen_certified_skills_lpba']
        self.assertEqual(sum(x['valid_deterministic_executions'] for phase in phases.values() for x in phase),
                         summary['lifecycle']['phase_summary']['qwen_certified_skills_lpba']['entire_lifecycle']['complete_workload']['valid_deterministic_executions'])
        self.assertEqual(sum(x['stale_or_invalid_deterministic_executions'] for phase in phases.values() for x in phase), 0)

    def test_completion_objective_and_independent_oracle_by_permutations(self):
        rng = random.Random(20261008)
        for repeat in range(20):
            n = rng.randrange(3, 7)
            items = tuple(WorkItem(str(i), Literal(str(i), ()), 'audit', rng.randrange(1, 6)) for i in range(4))
            supports = {item.item_id: tuple(sorted(set(rng.randrange(1, 1 << n) for _ in range(2)))) for item in items}
            cat = Catalog(tuple(str(i) for i in range(n)), tuple(float(rng.randrange(1, 4)) for _ in range(n)), items, supports)
            best = min(weighted_completion_cost(cat, order) for order in itertools.permutations(range(n)))
            independent = brute_force_optimal_latency(cat)
            self.assertTrue(math.isclose(best, independent), repeat)
            exact = exact_oracle(cat, (.25,.5), max_relevant=n)
            self.assertTrue(math.isclose(best, weighted_completion_cost(cat, exact.order)))
            for chooser in [lpba_order, exact_density_order, committed_bundle_order]:
                order = chooser(cat); latency = weighted_completion_cost(cat, order)
                self.assertGreaterEqual(latency + 1e-9, best)
                self.assertTrue(math.isclose(latency, sum(cat.costs) * (sum(x.weight for x in items) - feasible_auc(cat, order))))

    def test_completion_cost_is_invariant_to_unused_tail(self):
        items = (WorkItem('j', Literal('j', ()), 'audit', 3),)
        original = Catalog(('a','b'),(1.,2.), items, {'j':(3,)})
        padded = Catalog(('a','b','unused'),(1.,2.,1000.),items,{'j':(3,)})
        self.assertEqual(weighted_completion_cost(original,(0,1)), weighted_completion_cost(padded,(0,1,2)))
        self.assertNotEqual(feasible_auc(original,(0,1)), feasible_auc(padded,(0,1,2)))

    def test_review_rows_recompute_original_means_and_completion_metrics(self):
        d = json.loads((ROOT/'results/acquisition_review/results.json').read_text())
        self.assertEqual(d['independent_oracle_matches'],400)
        self.assertEqual(len(d['theories']),400)
        for cohort,modes in d['summaries'].items():
            old = json.loads((ROOT/'results'/cohort/'results.json').read_text())
            for mode, summary in modes.items():
                rows = [r for r in d['theories'] if r['cohort']==cohort and r['mode']==mode]
                self.assertEqual(len(rows),100)
                for policy,entry in summary.items():
                    for row in rows:
                        actual=row['policies'][policy]
                        self.assertTrue(math.isclose(actual['completion_cost'],row['total_cost']*(row['certifiable_weight']-actual['auc']),abs_tol=1e-8))
                    got=sum(r['policies'][policy]['relative_completion_excess'] for r in rows)/len(rows)
                    self.assertTrue(math.isclose(got,entry['relative_completion_excess']['mean'],abs_tol=1e-12))
                    if policy in old['studies'][mode]['summary']['mean_auc']:
                        self.assertTrue(math.isclose(entry['mean_auc'],old['studies'][mode]['summary']['mean_auc'][policy],abs_tol=1e-12))

    def test_runtime_fault_review_reproduces_all_cases(self):
        spec=importlib.util.spec_from_file_location('runtime_review', ROOT/'scripts/run_runtime_review.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        got=module.run();expected=json.loads((ROOT/'results/runtime_review/results.json').read_text())
        self.assertEqual(got,expected)
        self.assertEqual(got['case_count'],15)
        self.assertEqual(got['unsafe_write_effects'],0)
        regression=json.loads((ROOT/'results/runtime_review/regression.json').read_text())
        self.assertEqual(regression['before']['write_effects'],1)
        self.assertEqual(regression['after']['write_effects'],0)

    def test_paired_review_summary_regenerates(self):
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp)/'paired.json'
            subprocess.run([sys.executable,str(ROOT/'scripts/summarize_acquisition_review.py'),'--output',str(target)],check=True,stdout=subprocess.DEVNULL)
            self.assertEqual(target.read_bytes(),(ROOT/'results/acquisition_review/paired_summary.json').read_bytes())

    def test_crossref_bibliography_uses_issue_year_and_standard_fields(self):
        spec=importlib.util.spec_from_file_location('reference_generator', ROOT/'scripts/verify_references.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        metadata={'type':'journal-article','title':['Published article'],'author':[{'family':'Author','given':'A.'}],
                  'published':{'date-parts':[[2015]]},'published-print':{'date-parts':[[2017]]},
                  'DOI':'10.example/record','container-title':['Journal'],'volume':'77','issue':'3','page':'661-685'}
        entry=module.bib_from_crossref('example',metadata)
        self.assertIn('year = {2017}',entry)
        self.assertIn('number = {3}',entry)
        self.assertIn('pages = {661-685}',entry)
        self.assertNotIn('  page =',entry)
        self.assertNotIn('  issue =',entry)

if __name__ == '__main__': unittest.main()
