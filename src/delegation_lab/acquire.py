"""Matched acquisition baselines and LPBA's proof-support bundle scheduler."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from random import Random
from typing import Iterable, Mapping, Sequence

from .certify import HornProver
from .evidence import EvidenceKind, EvidenceRecord, EvidenceStore
from .ir import Literal


class AcquisitionMethod(str, Enum):
    RANDOM = "random"
    FREQUENCY_PER_COST = "frequency_per_cost"
    INFORMATION_GAIN_PER_COST = "information_gain_per_cost"
    WORKLOAD_WEIGHTED_DRD = "workload_weighted_drd_hec"
    ONE_STEP_VOI = "one_step_certification_voi"
    COUNTEREXAMPLE_FIRST = "counterexample_first_global_memoization"
    EAGER = "eager_full_formalization"
    LPBA = "lpba"
    LPBA_NO_BUNDLES = "lpba_no_bundles"
    LPBA_NO_SHARED_EVIDENCE = "lpba_no_shared_evidence"
    LPBA_NO_WORKLOAD_WEIGHTING = "lpba_no_workload_weighting"


@dataclass(frozen=True, slots=True)
class WorkItem:
    """A recurrent request whose target is known to the controller, not its label."""

    item_id: str
    target: Literal
    cluster_id: str
    weight: float = 1.0
    expected_outcome: str | None = None  # evaluation-only; never read by schedulers


@dataclass(frozen=True, slots=True)
class AcquisitionStep:
    step: int
    acquired: tuple[str, ...]
    cumulative_evidence_cost: float
    proof_checks: int
    certified_weight: float
    certified_items: int
    total_weight: float
    chosen_bundle: tuple[str, ...]
    score: float


@dataclass(slots=True)
class AcquisitionRun:
    method: AcquisitionMethod
    seed: int
    budget: float
    steps: list[AcquisitionStep] = field(default_factory=list)
    acquired: tuple[str, ...] = ()
    unresolved: tuple[str, ...] = ()
    total_proof_checks: int = 0

    @property
    def final(self) -> AcquisitionStep | None:
        return self.steps[-1] if self.steps else None

    @property
    def auc(self) -> float:
        """Trapezoidal certified-workload/cost area, normalized by budget."""
        if not self.steps or self.budget <= 0:
            return 0.0
        points = [(0.0, 0.0)] + [(step.cumulative_evidence_cost, step.certified_weight) for step in self.steps]
        area = sum(
            (right[0] - left[0]) * (left[1] + right[1]) / 2
            for left, right in zip(points, points[1:])
        )
        return area / self.budget

    def as_dict(self) -> dict:
        return {
            "method": self.method.value,
            "seed": self.seed,
            "budget": self.budget,
            "auc": self.auc,
            "acquired": list(self.acquired),
            "unresolved": list(self.unresolved),
            "total_proof_checks": self.total_proof_checks,
            "steps": [asdict(step) for step in self.steps],
        }


class AcquisitionEngine:
    """All methods see the exact same candidates, costs, prover, and workload.

    The implementation intentionally does not inspect ``expected_outcome`` or any
    benchmark proof DAG.  The formal-source mode uses source representations to
    compute candidate supports; it is therefore labeled an acquisition oracle,
    not a natural-language model result.
    """

    def __init__(
        self,
        candidates: Sequence[EvidenceRecord],
        workload: Sequence[WorkItem],
        *,
        max_bundle_candidates: int = 32,
        max_local_bundle_queries: int = 3,
        max_alternative_supports: int = 8,
    ) -> None:
        if len({candidate.id for candidate in candidates}) != len(candidates):
            raise ValueError("candidate evidence ids must be unique")
        self.candidates = tuple(sorted(candidates, key=lambda record: record.id))
        self.workload = tuple(workload)
        self.prover = HornProver(max_alternatives=max_alternative_supports)
        self.max_bundle_candidates = max_bundle_candidates
        self.max_local_bundle_queries = max_local_bundle_queries
        self._full_store = EvidenceStore(candidate.accepted() for candidate in self.candidates)
        self._full_supports = self._compute_full_supports()

    def run(
        self,
        method: AcquisitionMethod,
        *,
        budget: float,
        seed: int,
        initial_records: Sequence[EvidenceRecord] = (),
    ) -> AcquisitionRun:
        """Acquire from a trusted starting cache when supplied.

        Lifecycle maintenance uses this path to hold unaffected evidence fixed
        while selecting only revised dependencies to reacquire.  Initial
        records incur no cost in this invocation; callers account for their
        earlier acquisition separately.
        """
        store = EvidenceStore(record.accepted() for record in initial_records)
        unknown_initial = store.ids.difference(candidate.id for candidate in self.candidates)
        if unknown_initial:
            raise ValueError(f"initial records are outside this candidate universe: {sorted(unknown_initial)}")
        remaining = {candidate.id: candidate for candidate in self.candidates if candidate.id not in store.ids}
        rng = Random(seed)
        run = AcquisitionRun(method=method, seed=seed, budget=budget)
        cost = 0.0
        if method is AcquisitionMethod.EAGER:
            all_cost = sum(candidate.cost for candidate in remaining.values())
            if all_cost <= budget:
                purchased = tuple(sorted(remaining))
                for candidate in remaining.values():
                    store.admit(candidate.accepted())
                cost = all_cost
                certified, checks = self._certified(store, complete_scope=True)
                run.steps.append(
                    AcquisitionStep(1, purchased, cost, checks, self._weight(certified), len(certified), self._total_weight, purchased, 0.0)
                )
                run.total_proof_checks += checks
                remaining.clear()
            run.acquired = tuple(sorted(store.ids))
            run.unresolved = tuple(item.item_id for item in self.workload if item.item_id not in self._certified(store, complete_scope=False)[0])
            return run

        step_number = 0
        while remaining:
            choice = self._choose(method, store, remaining, rng)
            if choice is None:
                break
            record_id, bundle, score = choice
            candidate = remaining[record_id]
            if cost + candidate.cost > budget:
                affordable = [record for record in remaining.values() if cost + record.cost <= budget]
                if not affordable:
                    break
                # A planned bundle can be unaffordable while a smaller action remains.
                candidate = min(affordable, key=lambda record: (record.cost, record.id))
                record_id, bundle, score = candidate.id, (candidate.id,), 0.0
            store.admit(candidate.accepted())
            remaining.pop(record_id)
            cost += candidate.cost
            step_number += 1
            certified, checks = self._certified(store, complete_scope=not remaining)
            run.total_proof_checks += checks
            run.steps.append(
                AcquisitionStep(
                    step_number,
                    (record_id,),
                    cost,
                    checks,
                    self._weight(certified),
                    len(certified),
                    self._total_weight,
                    tuple(bundle),
                    score,
                )
            )
        certified, _ = self._certified(store, complete_scope=not remaining)
        run.acquired = tuple(sorted(store.ids))
        run.unresolved = tuple(item.item_id for item in self.workload if item.item_id not in certified)
        return run

    @property
    def _total_weight(self) -> float:
        return sum(item.weight for item in self.workload)

    def _compute_full_supports(self) -> dict[str, tuple[frozenset[str], ...]]:
        derivations = self.prover.all_derivations(self._full_store)
        result: dict[str, tuple[frozenset[str], ...]] = {}
        for item in self.workload:
            alternatives = list(derivations.get(item.target, ())) + list(derivations.get(item.target.negate(), ()))
            supports: list[frozenset[str]] = []
            for proof in alternatives:
                if proof.support not in supports:
                    supports.append(proof.support)
            result[item.item_id] = tuple(supports)
        return result

    def _certified(self, store: EvidenceStore, *, complete_scope: bool) -> tuple[set[str], int]:
        derivations = self.prover.all_derivations(store)
        certified: set[str] = set()
        checks = 0
        for item in self.workload:
            checks += 1
            if item.target in derivations or item.target.negate() in derivations:
                certified.add(item.item_id)
            # OWA Unknown is only certified after complete source closure, never
            # mistaken for controller deferral while evidence remains hidden.
            elif complete_scope and not self._full_supports[item.item_id]:
                certified.add(item.item_id)
        return certified, checks

    def _weight(self, item_ids: Iterable[str]) -> float:
        ids = set(item_ids)
        return sum(item.weight for item in self.workload if item.item_id in ids)

    def _choose(
        self,
        method: AcquisitionMethod,
        store: EvidenceStore,
        remaining: Mapping[str, EvidenceRecord],
        rng: Random,
    ) -> tuple[str, tuple[str, ...], float] | None:
        if method is AcquisitionMethod.RANDOM:
            record_id = rng.choice(sorted(remaining))
            return record_id, (record_id,), 0.0
        if method is AcquisitionMethod.FREQUENCY_PER_COST:
            scores = {record_id: self._frequency(record) / record.cost for record_id, record in remaining.items()}
            return self._best_single(scores)
        if method is AcquisitionMethod.INFORMATION_GAIN_PER_COST:
            scores = {record_id: self._information_score(record) / record.cost for record_id, record in remaining.items()}
            return self._best_single(scores)
        if method is AcquisitionMethod.WORKLOAD_WEIGHTED_DRD:
            return self._decision_directed(remaining, weighted=True)
        if method is AcquisitionMethod.COUNTEREXAMPLE_FIRST:
            return self._counterexample_first(remaining)
        if method in (AcquisitionMethod.ONE_STEP_VOI, AcquisitionMethod.LPBA_NO_BUNDLES):
            return self._one_step_voi(store, remaining)
        if method is AcquisitionMethod.LPBA_NO_SHARED_EVIDENCE:
            return self._lpba(remaining, use_bundles=True, shared_value=False, weighted=True)
        if method is AcquisitionMethod.LPBA_NO_WORKLOAD_WEIGHTING:
            return self._lpba(remaining, use_bundles=True, shared_value=True, weighted=False)
        if method is AcquisitionMethod.LPBA:
            return self._lpba(remaining, use_bundles=True, shared_value=True, weighted=True)
        raise ValueError(f"unsupported method {method}")

    def _best_single(self, scores: Mapping[str, float]) -> tuple[str, tuple[str, ...], float]:
        record_id = min(scores, key=lambda item: (-scores[item], item))
        return record_id, (record_id,), scores[record_id]

    def _frequency(self, record: EvidenceRecord) -> float:
        predicate_score = 0.0
        predicates = self._record_predicates(record)
        for item in self.workload:
            if item.target.predicate in predicates:
                predicate_score += item.weight
        return predicate_score

    def _information_score(self, record: EvidenceRecord) -> float:
        """A label-free structural information surrogate, deliberately not LPBA."""
        predicates = self._record_predicates(record)
        connected = 0.0
        for item in self.workload:
            if item.target.predicate in predicates:
                connected += 1.0
        # Rules joining several predicates are structurally more informative;
        # this baseline does not consider proof completion or joint acquisition.
        return connected * len(predicates)

    @staticmethod
    def _record_predicates(record: EvidenceRecord) -> set[str]:
        if record.kind is EvidenceKind.FACT and isinstance(record.payload, Literal):
            return {record.payload.predicate}
        if record.kind is EvidenceKind.RULE:
            return {record.payload.head.predicate, *(atom.predicate for atom in record.payload.body)}  # type: ignore[union-attr]
        return set()

    def _decision_directed(self, remaining: Mapping[str, EvidenceRecord], *, weighted: bool) -> tuple[str, tuple[str, ...], float]:
        scores = {record_id: 0.0 for record_id in remaining}
        acquired = set(self._full_store.ids).difference(remaining)
        for item in self.workload:
            for support in self._full_supports[item.item_id]:
                missing = support.difference(acquired)
                if not missing:
                    continue
                contribution = (item.weight if weighted else 1.0) / len(missing)
                for record_id in missing:
                    if record_id in scores:
                        scores[record_id] += contribution / remaining[record_id].cost
        return self._best_single(scores)

    def _counterexample_first(self, remaining: Mapping[str, EvidenceRecord]) -> tuple[str, tuple[str, ...], float]:
        # Memoized proof-obstruction frequency. Unlike LPBA, it asks one source at
        # a time and cannot receive credit for a jointly completed support.
        scores = {record_id: 0.0 for record_id in remaining}
        acquired = set(self._full_store.ids).difference(remaining)
        for item in self.workload:
            alternatives = [support.difference(acquired) for support in self._full_supports[item.item_id] if support.difference(acquired)]
            if not alternatives:
                continue
            obstruction = min(alternatives, key=lambda missing: (len(missing), tuple(sorted(missing))))
            for record_id in obstruction:
                if record_id in scores:
                    scores[record_id] += item.weight / remaining[record_id].cost
        return self._best_single(scores)

    def _one_step_voi(self, store: EvidenceStore, remaining: Mapping[str, EvidenceRecord]) -> tuple[str, tuple[str, ...], float]:
        before, _ = self._certified(store, complete_scope=False)
        before_weight = self._weight(before)
        scores: dict[str, float] = {}
        for record_id, record in remaining.items():
            trial = store.copy()
            trial.admit(record.accepted())
            after, _ = self._certified(trial, complete_scope=False)
            scores[record_id] = (self._weight(after) - before_weight) / record.cost
        return self._best_single(scores)

    def _lpba(
        self,
        remaining: Mapping[str, EvidenceRecord],
        *,
        use_bundles: bool,
        shared_value: bool,
        weighted: bool,
    ) -> tuple[str, tuple[str, ...], float]:
        acquired = set(self._full_store.ids).difference(remaining)
        # A bundle is scored by *new* certified workload. Counting a region
        # that is already certified gives every future purchase a constant
        # (and potentially large) free bonus, which is neither an acquisition
        # benefit nor a valid marginal bundle objective.
        already_certified = {
            item.item_id
            for item in self.workload
            if any(support.issubset(acquired) for support in self._full_supports[item.item_id])
        }
        candidate_bundles: set[frozenset[str]] = set()
        unfinished: list[tuple[WorkItem, frozenset[str]]] = []
        for item in self.workload:
            for support in self._full_supports[item.item_id]:
                missing = support.difference(acquired)
                if missing:
                    unfinished.append((item, missing))
                    # Complete support bundles are retained regardless of length;
                    # the global caller's budget, not the local cap, decides them.
                    candidate_bundles.add(frozenset(missing))
        if use_bundles:
            # Small unions reveal cross-certificate shared-evidence opportunities.
            for index, (_, left) in enumerate(unfinished):
                for _, right in unfinished[index + 1 :]:
                    union = left.union(right)
                    if len(union) <= self.max_local_bundle_queries:
                        candidate_bundles.add(frozenset(union))
                    if len(candidate_bundles) >= self.max_bundle_candidates * 4:
                        break
                if len(candidate_bundles) >= self.max_bundle_candidates * 4:
                    break
        else:
            candidate_bundles = {frozenset({record_id}) for record_id in remaining}

        ranked: list[tuple[float, tuple[str, ...]]] = []
        for bundle in candidate_bundles:
            bundle = bundle.intersection(remaining)
            if not bundle:
                continue
            cost = sum(remaining[record_id].cost for record_id in bundle)
            value = 0.0
            for item in self.workload:
                if item.item_id in already_certified:
                    continue
                supports = [support.difference(acquired) for support in self._full_supports[item.item_id]]
                completing = [missing for missing in supports if missing and missing.issubset(bundle)]
                if completing:
                    if shared_value:
                        value += item.weight if weighted else 1.0
                    else:
                        # Count a certificate's support separately: this removes
                        # shared coverage credit while retaining bundle completion.
                        value += sum((item.weight if weighted else 1.0) for _ in completing)
            ranked.append((value / cost if cost else 0.0, tuple(sorted(bundle))))
        if not ranked:
            return self._decision_directed(remaining, weighted=weighted)
        ranked.sort(key=lambda row: (-row[0], len(row[1]), row[1]))
        score, bundle = ranked[0]
        # Replanning executes only the first query. Prefer the member that appears
        # in most unfinished supports, then deterministic id/cost tie-breaking.
        influence = {
            record_id: sum(1 for _, missing in unfinished if record_id in missing) / remaining[record_id].cost
            for record_id in bundle
        }
        first = min(bundle, key=lambda record_id: (-influence[record_id], record_id))
        return first, bundle, score
