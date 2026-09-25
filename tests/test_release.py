from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from delegation_lab.acquire import WorkItem
from delegation_lab.ir import Literal
from delegation_lab.experiments.proofwriter_exact_core import Catalog, exact_oracle, feasible_auc, lpba_order


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

if __name__ == '__main__': unittest.main()
