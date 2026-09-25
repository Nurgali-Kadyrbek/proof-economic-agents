# Controlled telecom authority-lifecycle protocol

## Claim boundary

This is one controlled lifecycle comparison over the same pinned tau2 telecom
solo workload design. The telecom test split was consumed by the completed
stream result, so this replay is a **mechanism and lifecycle study**, not a
fresh held-out task-success claim and not an official tau2 leaderboard score.
It adds no benchmark, model, skill language, or model-training procedure.

The frozen model is `Qwen/Qwen3.5-2B` revision
`15852e8c16360a2fea060d615a32b45270f8a8fc`. Every arm uses the same task
order, operational-state reset, prompts, tools, 16-turn limit, and source
policy. The semantic portfolio and exact-decision cache persist within an arm
over the complete lifecycle.

## Workload phases and denominators

The lifecycle stream is predeclared as:

```text
v1 recurrence: existing telecom test episodes 1..40, pinned order
source-version event
v2 recurrence: the same existing telecom test episodes 1..40, same order
```

Thus the complete-workload denominator is 40 episodes in `pre_change`, 40 in
`post_change`, and 80 over the entire lifecycle. The predeclared eligible
recurrent predicate remains the source-state predicate in
[`recurrent_stream_protocol.md`](recurrent_stream_protocol.md); it is applied
independently to every phase and its actual denominator is included in every
reported table/JSON summary. It is not pooled with the previous 37-episode
post-acquisition steady-state diagnostic.

## Controlled source-version event

After pre-change episode 40 and before post-change episode 1, increment the
reviewed source version (without changing its payload) for precisely these
source-policy items:

```text
tau2-telecom-reviewed-contract:airplane-policy
tau2-telecom-reviewed-contract:airplane-proof-rule
```

Together these are the complete policy support for the bounded airplane-repair
write; read-network evidence and the mobile-data and Data Saver skills are not
changed. The airplane branch was chosen using source/train information only: it
has the largest predeclared initialized-train recurrence count (40, versus 28
for mobile data and 12 for Data Saver). This controlled event tests versioned
proof maintenance, not a claim about a natural policy-change rate.

Each evidence item is also a runtime source epoch for this study. The registry
suspends every active skill whose exact proof support includes a changed item
before the first v2 task. An active certificate is valid only if the skill
checker replays against current source versions; a stale active authority or a
stale/invalid deterministic commit is a failure.

## Arms and maintenance rules

| Arm | Initial authority | At v2 source event | Exact-cache treatment |
| --- | --- | --- | --- |
| Qwen | None | No semantic portfolio | N/A |
| Exact cache | None | No semantic portfolio | Cache persists. Its actual key contains messages and operational state version, not reviewed-source epochs; it is neither cleared nor granted source-version awareness. Cached proposals still execute through ordinary tools. |
| Eager certified delegation | All reviewed skills before v1 episode 1 | Suspend the affected airplane skill, review/reprove its changed source items immediately, then register fresh authority before v2 episode 1 | N/A |
| LPBA certified delegation | Proof-directed acquisition on matching v1 workload | Suspend only the affected airplane skill. Retain unaffected authority. Reacquire/reprove the missing support only on the first matching v2 request. | N/A |

Evidence/policy review is performed by the existing authoritative source
adapter; no LLM proposal is elevated to evidence. Eager is deliberately
allowed its strong all-semantics upfront policy. LPBA retains the existing
bundle-aware planner and selective registry maintenance.

## Measures

Each phase and population summary explicitly reports its population label and
episode denominator. The output records:

1. benchmark/action-contract success;
2. valid deterministic skill executions and valid useful guarded writes;
3. stale/invalid deterministic executions and stale active authority;
4. Qwen calls and tokens; exact-cache hits;
5. initial evidence/proof-review and revalidation/reacquisition costs;
6. authority downtime: post-event source-state repair obligations lacking
   currently valid authority at episode start;
7. cumulative valid useful deterministic workload, review cost, calls, and
   tokens across the 80-episode lifecycle.

There is no assumed dollar conversion. Break-even is reported for each LPBA
comparison as a sensitivity grid over review-unit-to-model-call values
`{0, 0.25, 0.5, 1, 2, 4}`; tokens remain separately reported rather than
silently monetized.

The primary question is whether LPBA produces more **valid** useful
deterministic workload per acquisition, inference, and maintenance cost over
the changed recurrent stream than exact cache and eager delegation. A negative
or tied result is retained as such.

## Freeze

Before test execution, `configs/tau2_telecom_lifecycle.yaml` records hashes of
the lifecycle runner and its core model-independent dependencies. The runner
verifies them before loading test tasks. The configuration SHA and freeze
manifest are appended here immediately before launch.

### Frozen before lifecycle replay — 2026-09-17

The corrected test configuration SHA-256 is
`8306854e6b5934a868f9ca81111295383a90ac851189b8be2942bea69e80a2d3`.
Its manifest verifies 15 source files before loading test tasks, including the
generic evidence, proof, runtime, lifecycle, skill, planner, cache/controller,
Qwen adapter, telecom adapter, original stream adapter, and this lifecycle
runner. A two-train-task mock-adapter lifecycle smoke completed with zero stale
active authority and invariant initial-state eligibility membership; it is not
an efficacy result.

The first lifecycle execution is archived as
`results/archive/tau2_telecom_lifecycle_pre_initial_state_eligibility/` and is
excluded from all analysis. It exposed a reporting-only implementation defect:
the new runner mistakenly evaluated the eligibility predicate after an arm had
altered device state, creating arm-dependent denominators. The corrective
change captures eligibility immediately after task initialization and asserts
identical phase membership across arms. No method, source-event selection,
source item, prompt, model, task schedule, turn limit, or score was altered;
the corrected freeze above was recorded before the replacement run.
