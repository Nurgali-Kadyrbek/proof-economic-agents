# Controlled ProofWriter workload-skew sensitivity

Frozen before evaluating new demand weights on the already selected 100
structural-stress theories. This is a **post-hoc sensitivity**, not a fresh
theory cohort or an estimate of real demand.

For each theory, rank question IDs by SHA-256 of `seed|item_id`, using seeds
11, 23, and 47. With $n$ questions and zero-based rank $r$, set
$z=2r/(n-1)-1$ (or zero if $n=1$), then set
$w=\exp(\alpha z)$ and normalize the theory's mean weight to 1. Evaluate
$\alpha\in\{0,0.75,1.5,3.0\}$; compute alpha-zero once because all seeds
coincide. This changes only workload weight, not source evidence, proof
supports, costs, query targets, or selected theory identities. Gold answers,
reference proofs, and policy outcomes do not determine weights.

Use the same complete minimal-support catalog, LPBA criterion, Lazy First-Use,
one-step certification gain, support-score heuristic, and exact acquisition
order oracle as the frozen stress study. The primary descriptive series is
the theory-mean relative LPBA AUC regret to exact versus observed workload
weight coefficient of variation. Also report paired LPBA-minus-control AUC.
Average each theory's three seeds before a 10,000-replicate bootstrap across
theories; the same theory remains the resampling unit. Report alpha-zero once.

This measures algorithm sensitivity to controlled demand skew under a fixed
formal-source oracle. It is not an external-validity result for natural agent
traffic and was designed after the stress-cohort result was seen.
