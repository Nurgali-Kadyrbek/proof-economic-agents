#!/usr/bin/env python3
"""Generate every reported manuscript quantity from frozen public artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median


def read(root: Path, name: str):
    return json.loads((root / 'results' / name).read_text())


def macro(name: str, value: str) -> str:
    assert name.isalpha() and name[0].isupper()
    return f'\\newcommand{{\\{name}}}{{{value}}}\n'


def table(label: str, caption: str, headers: list[str], rows: list[list[str]], spec: str) -> str:
    width = len(headers)
    if any(len(row) != width for row in rows):
        raise ValueError(f'{label}: row/header column mismatch')
    lines = [r'\begin{table}[H]', rf'\caption{{{caption}\label{{{label}}}}}', r'\centering\small',
             rf'\begin{{tabular}}{{{spec}}}', r'\toprule', ' & '.join(headers) + r' \\', r'\midrule']
    lines += [' & '.join(row) + r' \\' for row in rows]
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
    return '\n'.join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    root = args.root.resolve(); out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    fresh = read(root, 'proofwriter_exact_characterization/results.json')
    stress = read(root, 'proofwriter_structural_stress/results.json')
    skew = read(root, 'proofwriter_workload_skew_sensitivity/results.json')
    bank = read(root, 'banking_exact_separable/results.json')
    budget = read(root, 'banking_exact_separable/budget_conditioned.json')
    agent = read(root, 'tau2_telecom_stream_heldout/public_summary.json')
    assert fresh['schema_version'] == 'proofwriter-exact-characterization-v1'
    assert stress['schema_version'] == 'proofwriter-structural-stress-v1'
    assert bank['schema_version'] == 'banking-separable-exact-development-audit-v1'
    assert budget['catalog_digest'] == bank['source']['catalog_digest']
    fs = fresh['studies']['uniform']['summary']; ss = stress['studies']['uniform']['summary']
    assert len(fresh['studies']['uniform']['theories']) == fs['theory_count']
    assert len(stress['studies']['uniform']['theories']) == ss['theory_count']
    stream = agent['stream']['stream_summary']; life = agent['lifecycle']
    base = stream['qwen_only']; lpba = stream['qwen_certified_skills_lpba']; cache = stream['qwen_exact_cache']
    call_delta = base['model_calls'] - lpba['model_calls']
    pct = lambda x, p=2: f'{100*x:.{p}f}\\%'
    num = lambda x, p=3: f'{x:.{p}f}'
    m = []
    vals = {
        'FreshN': str(fs['theory_count']), 'StressN': str(ss['theory_count']),
        'FreshExact': num(fs['mean_auc']['exact']), 'StressExact': num(ss['mean_auc']['exact']),
        'FreshLpba': num(fs['mean_auc']['lpba']), 'StressLpba': num(ss['mean_auc']['lpba']),
        'FreshLazy': num(fs['mean_auc']['lazy_first_use']), 'StressLazy': num(ss['mean_auc']['lazy_first_use']),
        'FreshRegret': pct(fs['relative_lpba_gap_to_exact']['mean']),
        'StressRegret': pct(ss['relative_lpba_gap_to_exact']['mean']),
        'FreshRegretLow': pct(fs['relative_lpba_gap_to_exact']['ci95'][0]),
        'FreshRegretHigh': pct(fs['relative_lpba_gap_to_exact']['ci95'][1]),
        'StressRegretLow': pct(ss['relative_lpba_gap_to_exact']['ci95'][0]),
        'StressRegretHigh': pct(ss['relative_lpba_gap_to_exact']['ci95'][1]),
        'FreshExactMatches': str(fs['exact_lpba_match_count']), 'StressExactMatches': str(ss['exact_lpba_match_count']),
        'FreshLead': num(fs['lpba_minus_control_auc']['lazy_first_use']['mean']),
        'StressLead': num(ss['lpba_minus_control_auc']['lazy_first_use']['mean']),
        'FreshPoolStart': str(fresh['dataset']['start_index']),
        'StressPoolEligible': str(stress['pool_counts']['eligible']),
        'StressMedianEvidence': str(median(x['relevant_evidence_count'] for x in stress['studies']['uniform']['theories'])),
        'FreshMedianEvidence': str(median(x['relevant_evidence_count'] for x in fresh['studies']['uniform']['theories'])),
        'BankFamilies': str(bank['assumptions_verified']['family_count']),
        'BankDevTasks': str(bank['assumptions_verified']['development_task_count']),
        'BankTotalCost': str(int(bank['total_cost'])),
        'BankCoreCost': str(int(bank['common_core_cost'])),
        'BankExact': num(bank['exact_subset_dp_auc']),
        'BankLpba': num(bank['lpba_global_atomic_auc']),
        'BankRegret': pct(bank['lpba_relative_regret']),
        'BankValueSum': str(int(bank['sum_workload_weight'])),
        'TelecomTickets': str(base['episodes']),
        'TelecomBaseCalls': str(base['model_calls']),
        'TelecomLpbaCalls': str(lpba['model_calls']),
        'TelecomDisplacedCalls': str(call_delta),
        'TelecomDisplacement': pct(call_delta / base['model_calls'], 1),
        'TelecomBaseSuccess': str(base['benchmark_successes']),
        'TelecomLpbaSuccess': str(lpba['benchmark_successes']),
        'TelecomCacheCalls': str(cache['model_calls']),
        'TelecomUsefulSkills': str(int(lpba['delegated_useful_workload'])),
        'TelecomPriorSkills': str(int(agent['stream']['certificate_reuse']['reused_skill_invocations'])),
        'TelecomReviewCost': str(int(lpba['portfolio_total_cost'])),
        'LifecycleEpisodes': str(life['phase_summary']['qwen_certified_skills_lpba']['entire_lifecycle']['complete_workload']['denominator_episodes']),
        'LifecycleValid': str(life['phase_summary']['qwen_certified_skills_lpba']['entire_lifecycle']['complete_workload']['valid_deterministic_executions']),
        'LifecycleStale': str(life['phase_summary']['qwen_certified_skills_lpba']['entire_lifecycle']['complete_workload']['stale_or_invalid_deterministic_executions']),
        'SkewRegret': pct(skew['summaries']['3.0']['lpba_relative_gap']['mean']),
        'SkewWeightCv': num(skew['summaries']['3.0']['mean_observed_weight_cv'],2),
        'BootstrapCount': str(fs['relative_lpba_gap_to_exact']['bootstrap_repetitions']),
    }
    for k,v in vals.items(): m.append(macro(k,v))
    (out/'numbers.tex').write_text(''.join(m))
    method_names = {'exact':'Exact oracle','lpba':'LPBA','lazy_first_use':'Lazy First-Use',
                    'one_step_certification_voi':'One-step gain','workload_weighted_drd_hec':'Support-score heuristic'}
    auc_rows = []
    for key,label in method_names.items():
        auc_rows.append([label,num(fs['mean_auc'][key]),num(ss['mean_auc'][key])])
    auc_rows += [[r'LPBA mean per-theory regret',pct(fs['relative_lpba_gap_to_exact']['mean']),pct(ss['relative_lpba_gap_to_exact']['mean'])],
                 [r'LPBA regret, 95\% bootstrap CI',
                  '['+pct(fs['relative_lpba_gap_to_exact']['ci95'][0])+', '+pct(fs['relative_lpba_gap_to_exact']['ci95'][1])+']',
                  '['+pct(ss['relative_lpba_gap_to_exact']['ci95'][0])+', '+pct(ss['relative_lpba_gap_to_exact']['ci95'][1])+']']]
    (out/'table_proofwriter.tex').write_text(table('tab:proofwriter',
        'Budget-feasible whole-order area on two nonoverlapping ProofWriter cohorts from one release. Each column is a mean across theories; regret is the mean of per-theory normalized gaps.',
        ['Method / measure',f'Source order ($n={fs["theory_count"]}$)',f'Structural stress ($n={ss["theory_count"]}$)'],auc_rows,'lrr'))
    rows=[]
    for fraction in ['0.1','0.25','0.5','0.75','0.9']:
        x=ss['budget_frontiers'][fraction]
        rows.append([pct(float(fraction),0),num(x['exact_mean_weight'],2),
                     num(x['policy_mean_weight']['lpba'],2),num(x['policy_mean_weight']['lazy_first_use'],2),
                     num(x['policy_mean_weight']['one_step_certification_voi'],2)])
    (out/'table_budget.tex').write_text(table('tab:budget',
        'Structural-stress cohort: mean certified workload at fixed fractions of each theory\'s source cost. Policy entries are affordable prefixes of frozen orders, not budget-conditioned replanning.',
        ['Budget','Exact','LPBA','First-Use','One-step'],rows,'lrrrr'))
    rows=[]
    for fraction in ['0.1','0.25','0.5','0.75','0.9']:
        x=budget['budget_frontiers'][fraction]
        rows.append([pct(float(fraction),0),str(int(x['budget'])),str(int(x['exact_value'])),
                     str(int(x['budget_conditioned_lpba_value'])),str(int(x['budget_aware_ratio_value']))])
    (out/'table_banking.tex').write_text(table('tab:banking',
        'One banking development catalog: exact knapsack value and genuinely budget-conditioned controls. Review costs are declared units.',
        ['Budget','Units','Exact','LPBA','Feasible ratio'],rows,'lrrrr'))
    rows=[]
    for key,label in [('qwen_only','Qwen only'),('qwen_exact_cache','Exact cache'),('qwen_simple_delegation','Simple delegation'),('qwen_certified_skills_lpba','Checked LPBA')]:
        x=stream[key]
        rows.append([label,str(x['model_calls']),f'{x["benchmark_successes"]}/{x["episodes"]}',
                     str(int(x['delegated_useful_workload'])),str(int(x['portfolio_total_cost']))])
    (out/'table_telecom.tex').write_text(table('tab:telecom',
        'One pinned local telecom stream. Review units have no validated conversion to money or inference cost; success differences are not evidence of a general quality gain.',
        ['Arm','Model calls','Success','Useful skills','Review units'],rows,'lrrrr'))
    coords=' '.join(f'({skew["summaries"][a]["mean_observed_weight_cv"]:.3f},{100*skew["summaries"][a]["lpba_relative_gap"]["mean"]:.3f})' for a in ['0.0','0.75','1.5','3.0'])
    (out/'figure_skew.tex').write_text(r'''\begin{figure}[H]
\centering
\begin{tikzpicture}
\begin{axis}[width=0.80\textwidth,height=5.2cm,xlabel={Mean within-theory workload-weight CV},ylabel={Mean LPBA regret to exact (\%)},xmin=0,xmax=1.55,ymin=0,ymax=3.0,grid=major,legend pos=north east]
\addplot+[black,mark=square*,thick] coordinates {'''+coords+r'''};
\addlegendentry{Controlled hash-rank weights}
\end{axis}
\end{tikzpicture}
\caption{Post-hoc sensitivity on the same structural-stress support graphs; the nonzero settings average three fixed hash seeds per theory. This is synthetic demand, not observed traffic.\label{fig:skew}}
\end{figure}
''')
    print('generated',len(list(out.glob('*.tex'))),'files in',out)

if __name__ == '__main__': main()
