# Recurrent-workload thesis checkpoint (pre-test protocol)

Retail and airline test results are development evidence only. The next
confirmatory domain is the existing pinned tau2 telecom **solo** benchmark,
which exposes the benchmark's ticket, service tools, and device tools to every
arm. The 74 train episodes may be used for source-grounded program review and
frequency estimates. The 40 test episodes will be processed once, in their
existing split order, by all four arms. No test reward, reference action,
Qwen trajectory, or LPBA outcome may inform program design. During source
inspection, the first five test ID strings were displayed; no test ticket,
state, evaluator action, or outcome was opened. This metadata exposure is
reported rather than silently calling the split pristine.

## Population and eligibility fixed before test

The complete benchmark is all 40 telecom test tickets. The primary thesis
population is the following source-state predicate, applied without reading
evaluation criteria: the ticket requests connectivity repair (mobile data,
"No Service", or MMS), has a stated phone number, and the initialized device
or line state has at least one of these policy-bounded repair obligations:

1. airplane mode is on;
2. mobile data is off for a mobile-data or MMS ticket;
3. Data Saver is on for a mobile-data ticket.

Each obligation is a recurrent, structured, bounded side effect specified by
the telecom policy/tools. The predicate is defined here from the source policy,
tool schemas, and train initialization patterns; it does not use test success.
An eligible task can have additional faults outside these programs, so useful
certified repair workload and whole-task success are separate metrics. Report
both complete-test and eligible-subset quality/compute results.
Roaming enablement was considered from train/source data but excluded before
test: the source customer can own several lines, and its available read tools
do not yield a phone-to-line binding within this bounded program without a
larger loop or an unproved hidden-state selector.

## Acquisition and stream unit

CA(K) counts only checked repair *obligations*: a complete guarded read/repair
program with a task-relevant postcondition. A read alone has no CA. Train
initialization frequency estimates future recurrence; the observed test stream
does not update these weights. LPBA scores the expected marginal CA of complete
or shared evidence bundles per evidence-review, proof-check, and expected
maintenance cost. The eager control acquires and checks the same candidate
program portfolio before episode one. Exact cache stores exact state/history
decisions. All arms use identical frozen Qwen weights, prompts, 16 decision
turns, tools, initial states, episode order, and reward evaluation.

Formally, with predeclared repair regions `r`, train frequency `f_r`, useful
repair value `u_r = 1` per completed obligation, and a checked active program
with some complete proof support inside trusted `K`:

```text
CA_train(K) = sum_r f_r * u_r * 1[complete support and checked active program]
Delta(B | K) = E[CA_train(K + accepted answers to bundle B)] - CA_train(K)
LPBA score = Delta(B | K) / (evidence review + proof checking + expected maintenance)
```

The three train frequencies are 40 (airplane), 28 (mobile-data switch), and
12 (Data Saver); 54/74 initialized train tasks satisfy the eligibility
predicate. The source evidence in this experiment has verified deterministic
answers, so candidate acceptance probability is one. The planner interface
also supports uncertain accepted answers, without treating an LLM proposal as
trusted evidence. Post-run reporting additionally counts actual source repair
obligations certifiable online and actual guarded repairs executed; these test
counts are never fed back to the acquisition scheduler.

Evidence, valid programs, and the exact cache persist across the 40-episode
stream. Each episode starts from a fresh initialized benchmark environment.
Charge evidence and proof acquisition once at portfolio level. Charge actual
maintenance only when dependency invalidation/revalidation occurs (no synthetic
change is injected into this held-out stream). Per-episode amortization is
descriptive; do not bootstrap that shared charge over episodes. With only one
independent stream, no acquisition-economic confidence interval is estimable.
Paired task/family intervals are limited to quality, safety, and compute.

Primary outputs are: CA(K) versus cumulative evidence cost; total calls/tokens
displaced over the stream; task/contract success; invalid tool actions; portfolio
evidence, proof, and realized maintenance cost; and a break-even sensitivity
curve over explicit model-call/token value assumptions, never one assumed
money conversion. The primary comparison is Qwen versus Qwen + LPBA on the
eligible stream. A positive thesis result requires equal-or-better quality
and materially less repeated Qwen work and/or invalid actions, plus more
useful certified workload per portfolio cost than eager delegation. A success
rate increase is a stronger, optional result. No positive claim will be made
if the data do not support it.

Invalid tool actions are tool errors or toggles of the three certified device
switches in the wrong direction. Solo-mode text replies and multiple proposals
are counted separately as communication-protocol violations; they are not
mislabelled as tool actions. This narrow metric does not prove that every other
tool call obeyed all telecom policy clauses.

## Source pin and freeze

tau2 revision: `2174a603f6d014ef94473ffa95957f6ce27100db`.

```text
781296ae5419169cb64aac0c4fa2aa6129c78f3c1592f64abd80df6fbabb7042  data/tau2-bench/data/tau2/domains/telecom/main_policy_solo.md
b174d1f9705b5d7df49468daf99560a5633f86d23e10499a1166a7753706a412  data/tau2-bench/data/tau2/domains/telecom/tech_support_workflow_solo.md
388552434d1b8225e33e7958bdb6efccfe92b49e11d1838568d9b559514e9d01  data/tau2-bench/src/tau2/domains/telecom/tools.py
03fa751eeea3734313a3f5274223d42750fd5c1a28f2624265a678f271ac309d  data/tau2-bench/src/tau2/domains/telecom/user_tools.py
37e562e1ae3242577407e1303b1548bc64e7ea68e37d36173e6747990ceaf8a4  data/tau2-bench/data/tau2/domains/telecom/tasks.json
605b488bb9a6acb3c7f4505240a855fdc8681d09aadb16a8f38b2efcfc5c3aec  data/tau2-bench/data/tau2/domains/telecom/split_tasks.json
```

Implementation/configuration hashes and the train-only diagnostic will be
appended before the first test execution. Test execution must stop if any
source or frozen implementation hash differs.

### Freeze recorded before test execution — 2026-09-16

The train-only two-ticket Qwen diagnostic completed with benchmark reward
2/2 in all four arms. Qwen and exact cache used 13 model calls over that
stream; eager skills and LPBA used 7, executing three checked repair writes.
LPBA purchased 14 evidence units plus 4 proof units; eager purchased 22 plus
6. This is a workflow diagnostic, not a confirmatory result. Its raw run
predates a reporting-only correction that separates solo communication
violations from invalid tool calls; the corrected counters are frozen below.
The source-grounded mock replay and 22 unit tests passed after the correction.

The frozen test configuration SHA-256 is
`c7416c279f506b91410e6be53c90caf4b5c9b131b9592c600eecc976dabe904c`.
Its `freeze.sha256` section verifies 16 implementation files before test
tasks are loaded, including the generic planner, runtime, Qwen adapter,
telecom adapter, evaluator, and stream accounting. This document and the
configuration were completed before opening any test ticket or reward.
