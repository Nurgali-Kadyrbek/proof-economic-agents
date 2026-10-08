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
    lines = [r'\begin{table}[H]', rf'\caption{{{caption}\label{{{label}}}}}', r'\centering\small\renewcommand{\arraystretch}{1.08}',
             rf'\begin{{tabular}}{{{spec}}}', r'\toprule', ' & '.join(headers) + r' \\', r'\midrule']
    lines += [' & '.join(row) + r' \\' for row in rows]
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
    return '\n'.join(lines)


def points(values, fmt='.3f'):
    return ' '.join(f'({x:{fmt}},{y:{fmt}})' for x,y in values)


def regret_percent(theory):
    optimum = theory['exact']['auc']
    return max(0.0, 100 * (optimum - theory['policies']['lpba']['auc']) / optimum)


def quantile(values, q):
    ordered = sorted(values)
    index = (len(ordered)-1)*q
    low = int(index)
    return ordered[low] + (ordered[min(low+1,len(ordered)-1)]-ordered[low])*(index-low)


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
    trace = read(root, 'tau2_telecom_stream_heldout/public_trace.json')
    review = read(root, 'acquisition_review/results.json')
    paired = read(root, 'acquisition_review/paired_summary.json')
    runtime_review = read(root, 'runtime_review/results.json')
    assert fresh['schema_version'] == 'proofwriter-exact-characterization-v1'
    assert stress['schema_version'] == 'proofwriter-structural-stress-v1'
    assert bank['schema_version'] == 'banking-separable-exact-development-audit-v1'
    assert budget['catalog_digest'] == bank['source']['catalog_digest']
    assert trace['schema_version'] == 'public-episode-metrics-v1'
    fs = fresh['studies']['uniform']['summary']; ss = stress['studies']['uniform']['summary']
    assert len(fresh['studies']['uniform']['theories']) == fs['theory_count']
    assert len(stress['studies']['uniform']['theories']) == ss['theory_count']
    stream = agent['stream']['stream_summary']; life = agent['lifecycle']
    base = stream['qwen_only']; lpba = stream['qwen_certified_skills_lpba']; cache = stream['qwen_exact_cache']
    call_delta = base['model_calls'] - lpba['model_calls']
    assert sum(x['agent_model_calls'] for x in trace['stream_arms']['qwen_only']) == base['model_calls']
    assert sum(x['agent_model_calls'] for x in trace['stream_arms']['qwen_certified_skills_lpba']) == lpba['model_calls']
    ft = fresh['studies']['uniform']['theories']; st = stress['studies']['uniform']['theories']
    fg = [regret_percent(x) for x in ft]; sg = [regret_percent(x) for x in st]
    wins = lambda data: sum(x['policies']['lpba']['auc'] > x['policies']['lazy_first_use']['auc'] + 1e-8 for x in data)
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
        'FreshOracleCap': str(fresh['oracle']['max_relevant_evidence']),
        'StressOracleCap': str(stress['oracle']['max_relevant_evidence']),
        'SupportCap': str(fresh['oracle']['max_minimal_supports_per_literal']),
        'FreshMedianRegret': pct(quantile(fg,.5)/100),
        'StressMedianRegret': pct(quantile(sg,.5)/100),
        'FreshNinetiethRegret': pct(quantile(fg,.9)/100),
        'StressNinetiethRegret': pct(quantile(sg,.9)/100),
        'FreshWinCount': str(wins(ft)), 'StressWinCount': str(wins(st)),
        'StressMedianComplementarity': pct(median(x['features']['complementarity_fraction'] for x in st),0),
        'FreshMedianComplementarity': pct(median(x['features']['complementarity_fraction'] for x in ft),0),
        'BankTenthExact': str(int(budget['budget_frontiers']['0.1']['exact_value'])),
        'BankTenthLpba': str(int(budget['budget_frontiers']['0.1']['budget_conditioned_lpba_value'])),
        'BankEarlyBudget': str(int(budget['budget_frontiers']['0.1']['budget'])),
        'IndependentOracleMatches': str(review['independent_oracle_matches']),
        'RuntimeCases': str(runtime_review['case_count']),
        'RuntimeAdverseCases': str(runtime_review['adverse_case_count']),
        'RuntimeUnsafeWrites': str(runtime_review['unsafe_write_effects']),
    }
    for prefix, cohort in [('Fresh','proofwriter_exact_characterization'), ('Stress','proofwriter_structural_stress')]:
        for suffix, mode in [('', 'uniform'), ('Prior', 'train_predicate_frequency')]:
            sr = review['summaries'][cohort][mode]['lpba']
            vals[prefix+suffix+'LatencyExcess'] = pct(sr['relative_completion_excess']['mean'])
            vals[prefix+suffix+'DelayReduction'] = pct(sr['completion_reduction_vs_lazy']['mean'])
            vals[prefix+suffix+'AreaRegret'] = pct(sr['relative_area_regret']['mean'])
            comparison = paired['summaries'][cohort][mode]['exact_density']['lpba_completion_reduction']
            vals[prefix+suffix+'DensityReduction'] = pct(comparison['mean'])
            vals[prefix+suffix+'DensityReductionLow'] = pct(comparison['ci95'][0])
            vals[prefix+suffix+'DensityReductionHigh'] = pct(comparison['ci95'][1])
        vals[prefix+'AreaRetention'] = pct(1-review['summaries'][cohort]['uniform']['lpba']['relative_area_regret']['mean'])
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
    (out/'table_design.tex').write_text(table('tab:design',
        'Frozen study design and inferential unit. The two ProofWriter cohorts share one dataset release; banking and telecom each contain one catalog or stream.',
        ['Study','Source / selection','Unit','Primary endpoint'],[
            ['Source-order',f'ProofWriter test, depth five; {fs["theory_count"]} consecutive theories','Theory','Whole-order area'],
            ['Structural stress',f'ProofWriter test, depth five; {ss["theory_count"]} eligible theories','Theory','Whole-order area'],
            ['Banking regime',f'Pinned policy catalog; {bank["assumptions_verified"]["family_count"]} packets','Catalog','Exact area / budget value'],
            ['Telecom stream',f'Pinned agent stream; {base["episodes"]} tickets','Stream','Calls / checked use'],
            ['Version replay',f'Two phases of {base["episodes"]} tickets','Replay','Stale executions'],
        ],r'>{\raggedright\arraybackslash}p{2.5cm}>{\raggedright\arraybackslash}p{5cm}>{\raggedright\arraybackslash}p{1.5cm}>{\raggedright\arraybackslash}p{3.0cm}'))
    (out/'table_regimes.tex').write_text(table('tab:regimes',
        'Decision rule implied by the objective and proved or observed support structure. The LPBA row is an empirical finding on the evaluated exact-solvable cohorts, not a general approximation guarantee.',
        ['Condition','Operational objective','Acquisition rule','Basis'],[
            ['Zero-delay unique supports','Realized cost on encountered regions','Lazy First-Use','Proposition~\\ref{prop:firstuse}'],
            ['Common core, disjoint packets','Whole-order area','Descending $w_i/c_i$','Proposition~\\ref{prop:separable}'],
            ['Common core, disjoint packets','Authority at a fixed budget','Packet knapsack','Proposition~\\ref{prop:separable}'],
            ['Alternative or complementary supports','Whole-order area','Bounded proof-directed LPBA','Exact-oracle cohorts'],
        ],r'>{\raggedright\arraybackslash}p{3.1cm}>{\raggedright\arraybackslash}p{3.0cm}>{\raggedright\arraybackslash}p{3.0cm}>{\raggedright\arraybackslash}p{2.9cm}'))
    coords=' '.join(f'({skew["summaries"][a]["mean_observed_weight_cv"]:.3f},{100*skew["summaries"][a]["lpba_relative_gap"]["mean"]:.3f})' for a in ['0.0','0.75','1.5','3.0'])
    intervals='\n'.join(r'\draw[black,thick] (axis cs:'+f'{skew["summaries"][a]["mean_observed_weight_cv"]:.3f},{100*skew["summaries"][a]["lpba_relative_gap"]["ci95"][0]:.3f}) -- (axis cs:{skew["summaries"][a]["mean_observed_weight_cv"]:.3f},{100*skew["summaries"][a]["lpba_relative_gap"]["ci95"][1]:.3f});' for a in ['0.0','0.75','1.5','3.0'])
    (out/'figure_skew.tex').write_text(r'''\begin{figure}[H]
\centering
\begin{tikzpicture}
\begin{axis}[width=0.80\textwidth,height=5.2cm,xlabel={Mean within-theory workload-weight CV},ylabel={Mean LPBA regret to exact (\%)},xmin=0,xmax=1.55,ymin=0,ymax=3.0,grid=major,legend pos=north east]
\addplot+[black,mark=square*,thick] coordinates {'''+coords+r'''};
\addlegendentry{Controlled hash-rank weights}
'''+intervals+r'''
\end{axis}
\end{tikzpicture}
\caption{Post-hoc sensitivity on the same structural-stress support graphs; vertical bars show theory-bootstrap 95\% intervals. Nonzero settings average three fixed hash seeds per theory. This is synthetic demand, not observed traffic.\label{fig:skew}}
\end{figure}
''')
    # Each figure is a complete float; all plotted numerical coordinates come from frozen artifacts.
    ecdf=[]
    for vals in (fg,sg):
        ordered=sorted(vals)
        ecdf.append(points([(v,(i+1)/len(ordered)) for i,v in enumerate(ordered)]))
    (out/'figure_regret_ecdf.tex').write_text(r'''\begin{figure}[H]
\centering
\begin{tikzpicture}
\begin{axis}[width=.89\textwidth,height=6.0cm,xlabel={Per-theory LPBA regret to exact oracle (\%)},ylabel={Fraction of theories at or below regret},xmin=0,xmax=12,ymin=0,ymax=1.02,grid=major,legend pos=south east]
\addplot+[black,mark=none,thick,const plot] coordinates {'''+ecdf[0]+r'''};
\addlegendentry{Source-order cohort}
\addplot+[black,dashed,mark=none,thick,const plot] coordinates {'''+ecdf[1]+r'''};
\addlegendentry{Structural-stress cohort}
\end{axis}
\end{tikzpicture}
\caption{Empirical distribution across independent theories within each nonoverlapping cohort. A point at zero means LPBA matches the exact whole-order value; the structural-stress cohort was selected on support structure, not sampled as a population estimate.\label{fig:regret}}
\end{figure}
''')
    frontiers=[]
    for summary in (fs,ss):
        curves=[]
        for policy in ('exact','lpba','lazy_first_use','one_step_certification_voi'):
            curves.append(points([(100*float(f), x['exact_mean_weight'] if policy=='exact' else x['policy_mean_weight'][policy])
                                  for f,x in sorted(summary['budget_frontiers'].items(),key=lambda z:float(z[0]))]))
        frontiers.append(curves)
    axisopts=r'width=.48\textwidth,height=5.1cm,xlabel={Budget (\% of source cost)},ylabel={Mean certified workload},xmin=5,xmax=95,grid=major,legend style={font=\scriptsize},legend pos=south east'
    lines=[]
    styles=('black,solid,mark=*','black,dashed,mark=square*','black,dotted,mark=triangle*','black,dashdotted,mark=diamond*')
    labels=('Exact','LPBA','First-Use','One-step')
    for title,curves in zip(('Source order','Structural stress'),frontiers):
        lines.append(r'\begin{axis}['+axisopts+',title={'+title+r'}]')
        for style,label,coord in zip(styles,labels,curves):
            lines.append(r'\addplot+['+style+'] coordinates {'+coord+r'};')
            lines.append(r'\addlegendentry{'+label+'}')
        lines.append(r'\end{axis}')
    (out/'figure_budget_frontiers.tex').write_text(r'''\begin{figure}[H]
\centering
\begin{tikzpicture}
'''+r'\begin{scope}[xshift=-.25\textwidth]'+'\n'.join(lines[:10])+r'\end{scope}'+'\n'+r'\begin{scope}[xshift=.25\textwidth]'+'\n'.join(lines[10:])+r'\end{scope}'+r'''
\end{tikzpicture}
\caption{Fixed-order affordable-prefix performance at five predefined budget fractions. Each point averages the same theories as the area analysis. Curves connect sampled budgets for visual guidance; no intervening budget values are claimed.\label{fig:budget}}
\end{figure}
''')
    # Ratio order and observed LPBA atomic purchases on the single separable catalog.
    ratio_steps=[(0.0,0.0),(bank['common_core_cost'],0.0)]
    paid=bank['common_core_cost']; certified=0.0
    for family in bank['ratio_order']:
        paid+=bank['packet_cost'][family]; certified+=bank['packet_value'][family]
        ratio_steps.append((paid,certified))
    lpba_steps=[(0.0,0.0)]; paid=0.0; certified_families=set()
    for step in bank['lpba_global_steps']:
        paid+=step['cost']; certified_families.update(step['regions'])
        lpba_steps.append((paid,sum(bank['packet_value'][i] for i in certified_families)))
    assert round(paid,8)==round(bank['total_cost'],8)
    bank_frontier=[]
    for key,x in sorted(budget['budget_frontiers'].items(),key=lambda z:float(z[0])):
        bank_frontier.append((x['budget'],x['exact_value'],x['budget_conditioned_lpba_value']))
    (out/'figure_banking.tex').write_text(r'''\begin{figure}[H]
\centering
\begin{tikzpicture}
\begin{axis}[width=.86\textwidth,height=5.5cm,xlabel={Cumulative declared review cost},ylabel={Certified development workload},xmin=0,xmax=370,ymin=0,ymax=280,grid=major,legend pos=south east]
\addplot+[black,thick,mark=none,const plot] coordinates {'''+points(ratio_steps,'.1f')+r'''};
\addlegendentry{Exact ratio order}
\addplot+[black,dashed,thick,mark=none,const plot] coordinates {'''+points(lpba_steps,'.1f')+r'''};
\addlegendentry{LPBA global order}
\addplot+[only marks,mark=*,black] coordinates {'''+points([(x,y) for x,y,_ in bank_frontier],'.1f')+r'''};
\addlegendentry{Exact hard-budget values}
\addplot+[only marks,mark=x,black] coordinates {'''+points([(x,z) for x,_,z in bank_frontier],'.1f')+r'''};
\addlegendentry{Budget-conditioned LPBA}
\end{axis}
\end{tikzpicture}
\caption{One document-derived banking catalog. Step lines show completed-packet authority in whole-order schedules; isolated markers are separately optimized hard-budget outcomes. The connected lines do not interpolate the fixed-budget optima.\label{fig:bank}}
\end{figure}
''')
    arms=[('qwen_only','Qwen only','black,solid,mark=none'),('qwen_exact_cache','Exact cache','black,dotted,mark=none'),
          ('qwen_simple_delegation','Simple delegation','black,dashdotted,mark=none'),
          ('qwen_certified_skills_lpba','Checked LPBA','black,dashed,mark=none')]
    callplots=[]
    for key,label,style in arms:
        total=0; series=[(0,0)]
        for row in trace['stream_arms'][key]:
            total+=row['agent_model_calls'];series.append((row['episode'],total))
        callplots += [r'\addplot+['+style+'] coordinates {'+points(series,'.0f')+r'};',r'\addlegendentry{'+label+'}']
        assert total==stream[key]['model_calls']
    (out/'figure_telecom_calls.tex').write_text(r'''\begin{figure}[H]
\centering
\begin{tikzpicture}
\begin{axis}[width=.86\textwidth,height=6cm,xlabel={Completed telecom episodes},ylabel={Cumulative agent-model calls},xmin=0,xmax=40,ymin=0,grid=major,legend pos=north west,legend style={font=\scriptsize}]
'''+ '\n'.join(callplots)+r'''
\end{axis}
\end{tikzpicture}
\caption{One fixed-order telecom stream with a persistent portfolio per arm. Episode-level metrics are shipped without task text or dialogue. The exact cache is a strong call-count control, and the graph does not measure monetary or total compute cost.\label{fig:telecom}}
\end{figure}
''')
    phases=trace['lifecycle_arms']['qwen_certified_skills_lpba']
    before=phases['pre_change']; after=phases['post_change']
    active=[(0,0)]+[(row['episode'],row['active_programs_at_start']) for row in before]+[(40+row['episode'],row['active_programs_at_start']) for row in after]
    running=0; valid=[(0,0)]
    for i,row in enumerate(before+after,1):
        running+=row['valid_deterministic_executions'];valid.append((i,running))
    assert running==life['phase_summary']['qwen_certified_skills_lpba']['entire_lifecycle']['complete_workload']['valid_deterministic_executions']
    (out/'figure_lifecycle.tex').write_text(r'''\begin{figure}[H]
\centering
\begin{tikzpicture}
\begin{axis}[width=.86\textwidth,height=5.5cm,xlabel={Replay episode (pre-change then post-change)},ylabel={Active certificates at episode start},xmin=0,xmax=80,ymin=0,ymax=4,ytick={0,1,2,3,4},grid=major,axis y line*=left,legend pos=north west]
\addplot+[black,mark=none,thick,const plot] coordinates {'''+points(active,'.0f')+r'''};
\addlegendentry{Active certificates}
\draw[black,dashed] (axis cs:40.5,0) -- (axis cs:40.5,4);
\node[anchor=north west,font=\scriptsize] at (axis cs:41,3.9) {version event};
\end{axis}
\begin{axis}[width=.86\textwidth,height=5.5cm,xmin=0,xmax=80,ymin=0,ymax=105,ylabel={Cumulative valid executions},axis y line*=right,axis x line=none,ytick={0,25,50,75,100},legend pos=south east]
\addplot+[black,dashed,mark=none,thick] coordinates {'''+points(valid,'.0f')+r'''};
\addlegendentry{Valid executions}
\end{axis}
\end{tikzpicture}
\caption{Controlled metadata-version replay in the checked LPBA arm. One affected certificate is absent at the first post-change episode and reappears after revalidation; unaffected authority persists. Valid executions accumulate across both phases; stale deterministic executions are reported in the text. The source payload itself was unchanged.\label{fig:lifecycle}}
\end{figure}
''')
    (out/'figure_support_structure.tex').write_text(r'''\begin{figure}[H]
\centering
\begin{tikzpicture}[font=\small,source/.style={draw,rounded corners,minimum width=7mm,minimum height=5mm},proof/.style={draw,circle,inner sep=1.5pt},goal/.style={draw,rounded corners,fill=black!8,minimum width=8mm},edge/.style={->,>=stealth,thin}]
\node[anchor=west,font=\bfseries] at (0,2.0) {(a) Alternative supports};
\node[source] (a) at (.35,1) {$a$}; \node[source] (b) at (.35,0) {$b$}; \node[source] (c) at (.35,-1) {$c$};
\node[proof] (p) at (2,1) {$P_1$}; \node[proof] (q) at (2,-1) {$P_2$}; \node[goal] (j) at (3.8,0) {$j$};
\draw[edge] (a)--(p);\draw[edge] (b)--(p);\draw[edge] (a)--(q);\draw[edge] (c)--(q);\draw[edge] (p)--(j);\draw[edge] (q)--(j);
\node[anchor=west,font=\bfseries] at (5,2.0) {(b) Core and disjoint packets};
\node[source] (k) at (5.5,.8) {$K_0$};\node[source] (r) at (6.8,1.2) {$P_1$};\node[source] (s) at (6.8,0) {$P_2$};\node[source] (t) at (6.8,-1.2) {$P_3$};
\node[goal] (u) at (8.6,1.2) {$j_1$};\node[goal] (v) at (8.6,0) {$j_2$};\node[goal] (w) at (8.6,-1.2) {$j_3$};
\draw[edge] (k)--(r);\draw[edge] (k)--(s);\draw[edge] (k)--(t);\draw[edge] (r)--(u);\draw[edge] (s)--(v);\draw[edge] (t)--(w);
\end{tikzpicture}
\caption{Support topology determines acquisition behavior. In (a), the arrow pairs into each proof node mean conjunction, while the two proof nodes are alternatives for authorizing $j$; $a$ is shared. In (b), each region needs the common core and one disjoint packet. The drawing is schematic, not a measured instance.\label{fig:support}}
\end{figure}
''')
    (out/'figure_system.tex').write_text(r'''\begin{figure}[H]
\centering
\begin{tikzpicture}[font=\small,box/.style={draw,rounded corners,align=center,minimum height=8mm,text width=22mm},arrow/.style={->,>=stealth,thick},node distance=4mm]
\node[box] (source) {Versioned\\reviewed sources};
\node[box,right=of source] (support) {Complete\\proof support};
\node[box,right=of support] (check) {Checker and\\certificate};
\node[box,right=of check] (registry) {Persistent\\authority registry};
\node[box,right=of registry] (runtime) {Guarded\\tool execution};
\draw[arrow] (source)--(support);\draw[arrow] (support)--(check);\draw[arrow] (check)--(registry);\draw[arrow] (registry)--(runtime);
\node[box,below=9mm of registry] (event) {Source version\\event};
\draw[arrow] (event)--node[right,font=\scriptsize] {suspend affected} (registry);
\end{tikzpicture}
\caption{Operational path from reviewed semantics to deterministic action. A certificate records source dependencies; the registry checks versions and request-state guards before execution. A changed dependency suspends affected authority until revalidated.\label{fig:system}}
\end{figure}
''')
    completion_rows = []
    review_names = {'lpba':'LPBA', 'lazy_first_use':'Lazy First-Use',
                    'one_step_certification_voi':'One-step gain', 'workload_weighted_drd_hec':'Support score',
                    'committed_bundle':'Committed bundle', 'exact_density':'Exact-density greedy'}
    for key, label in review_names.items():
        values = []
        for cohort in ['proofwriter_exact_characterization','proofwriter_structural_stress']:
            entry = review['summaries'][cohort]['uniform'][key]['relative_completion_excess']
            values.append(f"{100*entry['mean']:.2f} [{100*entry['ci95'][0]:.2f}, {100*entry['ci95'][1]:.2f}]")
        completion_rows.append([label, *values])
    (out/'table_completion.tex').write_text(table('tab:completion',
        'Mean per-theory excess weighted first-certification cost over the independently recomputed exact optimum (percent; descriptive theory-bootstrap intervals). Unprovable work is excluded consistently. The source-order and stress columns use the original frozen uniform workloads.',
        ['Policy','Source order','Structural stress'],completion_rows,'lrr'))
    plots=[]
    keys = ['lazy_first_use','one_step_certification_voi','workload_weighted_drd_hec','committed_bundle','exact_density','lpba']
    for style,label,cohort in [('black,fill=black!15','Source order','proofwriter_exact_characterization'),
                              ('black,fill=white,postaction={pattern=north east lines}','Structural stress','proofwriter_structural_stress')]:
        coords=[]
        for i,key in enumerate(keys,1):
            s = review['summaries'][cohort]['uniform'][key]['relative_completion_excess']
            coords.append(f"({i},{100*s['mean']:.4f}) += (0,{100*(s['ci95'][1]-s['mean']):.4f}) -= (0,{100*(s['mean']-s['ci95'][0]):.4f})")
        plots.extend([r'\addplot+['+style+r',error bars/.cd,y dir=both,y explicit] coordinates {'+' '.join(coords)+r'};',
                      r'\addlegendentry{'+label+'}'])
    (out/'figure_completion.tex').write_text(r'''\begin{figure}[H]
\centering
\begin{tikzpicture}
\begin{axis}[ybar,bar width=8pt,width=.86\textwidth,height=6cm,ylabel={Excess weighted completion cost (\%)},ymin=0,xtick={1,2,3,4,5,6},xticklabels={First-Use,One-step,Support score,Committed,Density,LPBA},x tick label style={rotate=20,anchor=east,font=\scriptsize},legend pos=north east,legend style={font=\scriptsize},grid=major]
'''+ '\n'.join(plots)+r'''
\end{axis}
\end{tikzpicture}
\caption{Scheduling quality measured by weighted first-certification cost. Bars and theory-bootstrap intervals use the same frozen uniform cohorts. This measure is unchanged by adding unused source items after all certifiable work is acquired; it addresses possible dilution of relative area regret by a full-value tail.\label{fig:completion}}
\end{figure}
''')
    runtime_labels = {
        'nominal':'Unchanged reviewed program', 'semantic_edit':'Changed source meaning; unchanged version',
        'version_edit':'Changed source version', 'tool_tamper':'Changed tool program',
        'argument_tamper':'Changed program arguments', 'principal_mismatch':'Request principal mismatch',
        'missing_guard':'Missing write precondition', 'missing_post':'Failed read postcondition',
        'state_race':'State change before commit', 'policy_race':'Policy-only change before commit',
        'effect_type':'Tool effect classification mismatch', 'closure_invalidation':'Closure dependency event',
        'selective_invalidation':'Disjoint certificate under selective suspension', 'untrusted_proposal':'Unvalidated model proposal',
        'initial_snapshot_race':'Source update at initial version snapshot'}
    runtime_rows = [[runtime_labels[c['case']], 'Pass' if c['passed'] else 'Fail', str(c['write_effects'])] for c in runtime_review['cases']]
    (out/'table_runtime_review.tex').write_text(table('tab:runtime-review',
        'Deterministic runtime audit using constructed reviewed contracts. The unchanged case permits the intended write; adverse cases produce no write effects. Selective suspension also preserves the unrelated certificate. These cases check mechanisms, rather than estimating a population failure rate.',
        ['Injected condition','Expected behavior','Write effects'],runtime_rows,'lcc'))
    print('generated',len(list(out.glob('*.tex'))),'files in',out)

if __name__ == '__main__': main()
