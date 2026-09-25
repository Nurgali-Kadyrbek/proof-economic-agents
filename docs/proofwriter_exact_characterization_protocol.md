# Exact acquisition characterization protocol

Frozen before evaluating source-order ProofWriter depth-5 test theories
101–200 (zero-based slice `100:200`). The prior study used the first 100.
This is a new, nonoverlapping test cohort from the same published release,
not an independent benchmark or natural-language semantic evaluation.

## Question and metric

Each accepted formal source item has its existing declared review cost. A
work item becomes certified when **all** items of at least one minimal Horn
proof support have been acquired. The primary metric is budget-feasible AUC:
the integral of certified workload over paid evidence cost, divided by the
total cost of the theory's source items. Value is credited only after the
entire purchase is paid. Unknown-answer closure contributes at the terminal
full-source point only, which has zero area.

The exact oracle enumerates all inclusion-minimal proof supports from the
original formal source, then uses subset dynamic programming to find the
best acquisition order for that objective. It is an offline upper bound: it
sees the full formal source and future workload, like the earlier formal-source
oracle, and it does not represent an executable online agent. LPBA and the
controls are evaluated on the same support semantics and costs. No gold
answer or benchmark proof DAG enters any acquisition policy.

The exact subcohort is defined *only* by tractability: at most 20 distinct
source items occur in any minimal support. Every excluded theory and any
support-enumeration limit breach is counted and reported. The primary
analysis is the theory-cluster mean of `(optimum AUC − LPBA AUC) / optimum
AUC` on exact-solvable theories with positive optimum AUC. Also report exact
matches, worst gap, and a paired theory-bootstrap 95% interval. No claim of
near-optimality is made without seeing these results.

At budgets of 10%, 25%, 50%, 75%, and 90% of total source cost, compare
each policy's achieved certified workload with the exact knapsack frontier.
Policy values use the affordable prefix of its fixed acquisition order;
items later in that order are not skipped to fill a remaining budget gap.
Secondary controls are Lazy First-Use in source question order, one-step
certification value per cost, the repository's decision-directed support
heuristic, and eager acquisition. Two workload definitions are evaluated:
uniform weights and the existing train-question predicate prior saved in the
frozen `results/proofwriter_train_prior/results.json` artifact. They reuse
the same 100 theories and are not independent confirmations.

Report complementarity as the weighted fraction of certifiable work items
whose shortest minimal support uses at least two source items; shared
evidence as the fraction of relevant source cost appearing in supports of at
least two distinct work items; workload skew as the coefficient of variation
of work-item weights. Report associations with oracle gap descriptively,
without searching for a favorable threshold.

## Comparator scope

The published HEC/DRD problem assumes a prior over unknown hypotheses and
selects tests whose **outcomes** update a version space. Here the formal
source is exposed before scheduling; acquiring an item pays for checked
authority, not information about an unknown outcome. A faithful HEC policy
therefore has no nondegenerate direct mapping to this experiment. The
repository's prior `workload_weighted_drd_hec` is retained under its honest
name as an implemented support-scoring heuristic, never labeled canonical
HEC. The exact oracle is the stronger same-objective comparator.

The intended practical transfer is a policy-bounded recurrent agent workflow:
reviewed source items and tool contracts become reusable checked authority;
source-version changes suspend affected authority until revalidated. This
experiment measures scheduling headroom, not agent task success or semantic
translation accuracy. The existing telecom/lifecycle results supply the
separate operational evidence.
