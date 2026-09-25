# Outcome-blind ProofWriter structural stress protocol

Frozen before opening depth-5 test source-order indices `200:947`. The
original study used `0:100`; the exact-characterization confirmation used
`100:200`. This stress cohort is independent of those theory identities but
comes from the same synthetic ProofWriter release.

Selection uses **only the formal source and question predicates**, never
gold answers, reference proof DAGs, policy acquisitions, or outcome scores.
For each theory, enumerate inclusion-minimal Horn supports with the frozen
exact-support algorithm. An eligible theory must have at most 18 relevant
source items, positive weighted fraction of items requiring a multi-source
support, and positive fraction of relevant source cost shared by at least two
work items. Sort eligible theories by descending relevant source-item count,
then descending number of minimal supports, then ascending theory ID; use
the first 100. If fewer than 100 qualify, use all and report the count. Every
support-enumeration limit breach is excluded and reported.

Run the same LPBA complete-support criterion, Lazy First-Use, one-step VOI,
decision-directed support-score heuristic, eager acquisition, and exact
subset oracle as in the [source-order characterization](proofwriter_exact_characterization_protocol.md).
The primary outcome is theory-mean relative LPBA AUC regret to the exact
oracle, with a 10,000-replicate theory bootstrap interval. Report the
matched-budget frontier and pairwise AUC differences against the controls.
Uniform and frozen train-predicate-prior workloads reuse the same selected
theories and are robustness conditions, not independent confirmations.

This deliberately emphasizes complementary and shared proof supports; it is
not an estimate of the average ProofWriter test-theory performance. A
positive result would establish a stronger formal-source acquisition
characterization, not natural-language semantic correctness or agent task
success. Canonical HEC remains a different Bayesian test-outcome problem.
