"""Proof-directed acquisition of useful workload regions, not tool calls.

The model-independent certifier returns program IDs it can actually check
under a trial trusted evidence store. Only checked, useful regions count in
CA(K). Acquisition economics belong to the persistent workload stream.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable, Iterable, Literal, Mapping

from .evidence import EvidenceRecord, EvidenceStore, TrustLevel


@dataclass(frozen=True, slots=True)
class SkillOpportunity:
    """One useful recurrent region, possibly unlocked by alternative bundles.

    ``useful_value_per_occurrence`` is the predeclared fraction of a task
    contract completed by the program. Exploratory reads have zero unless
    they themselves complete a meaningful checked workload outcome.
    """

    region_id: str
    support_options: tuple[frozenset[str], ...]
    required_programs: frozenset[str]
    workload_frequency: float
    useful_value_per_occurrence: float
    expected_model_calls_displaced: float
    proof_cost: float
    maintenance_cost: float

    def __post_init__(self) -> None:
        if not self.support_options or any(not support for support in self.support_options):
            raise ValueError("every workload region needs a nonempty proof support")
        if not self.required_programs:
            raise ValueError("a workload region needs a checked program")
        if any(value < 0 for value in (
            self.workload_frequency, self.useful_value_per_occurrence,
            self.expected_model_calls_displaced, self.proof_cost,
            self.maintenance_cost,
        )):
            raise ValueError("workload values and costs must be nonnegative")

    @property
    def useful_value(self) -> float:
        return self.workload_frequency * self.useful_value_per_occurrence

    @property
    def displaced_calls(self) -> float:
        return self.workload_frequency * self.expected_model_calls_displaced


@dataclass(frozen=True, slots=True)
class CertifiedWorkloadValue:
    ca: float
    regions: tuple[str, ...]
    expected_model_calls_displaced: float


@dataclass(frozen=True, slots=True)
class BundleChoice:
    evidence_ids: tuple[str, ...]
    unlocked_regions: tuple[str, ...]
    expected_delta_ca: float
    expected_model_calls_displaced: float
    evidence_cost: float
    proof_cost: float
    maintenance_cost: float
    score: float


@dataclass(frozen=True, slots=True)
class AcquireOrDeferDecision:
    """A model-independent purchase-or-wait decision.

    ``defer`` is a real action: it reserves neither evidence nor budget and
    leaves the request to the caller's ordinary LLM fallback.  Scores remain
    in workload-value-per-declared-cost units; this class intentionally does
    not manufacture a currency conversion between review units and model
    tokens/calls.
    """

    action: Literal["acquire", "defer"]
    choice: BundleChoice | None
    effective_score: float
    expected_ready_ca: float
    expected_ready_calls_displaced: float
    reason: str


class SkillBundlePlanner:
    """Maximize expected marginal CA per evidence, proof, and maintenance cost.

    CA(K) = sum_r frequency(r) * useful_value(r) when some proof support of r
    is in trusted K *and* all programs needed by r pass the certifier.
    Acceptance probabilities are independent in this small interface and
    default to one for independently validated source evidence.
    """

    def __init__(
        self,
        candidates: Iterable[EvidenceRecord],
        opportunities: Iterable[SkillOpportunity],
        *,
        certify_programs: Callable[[EvidenceStore], Iterable[str]],
        acceptance_probabilities: Mapping[str, float] | None = None,
    ) -> None:
        records = tuple(candidates)
        self.candidates: Mapping[str, EvidenceRecord] = {record.id: record for record in records}
        if len(self.candidates) != len(records):
            raise ValueError("candidate evidence IDs must be unique")
        if any(record.trust is not TrustLevel.ACCEPTED or record.provenance.is_model_proposal for record in records):
            raise ValueError("candidate evidence must be independently accepted")
        self.opportunities = tuple(opportunities)
        if len({item.region_id for item in self.opportunities}) != len(self.opportunities):
            raise ValueError("workload region IDs must be unique")
        if any(not support.issubset(self.candidates) for item in self.opportunities for support in item.support_options):
            raise ValueError("a region support is outside the candidate evidence universe")
        self.certify_programs = certify_programs
        self.acceptance_probabilities = dict(acceptance_probabilities or {})
        if any(key not in self.candidates or not 0 <= value <= 1 for key, value in self.acceptance_probabilities.items()):
            raise ValueError("invalid acceptance probability")
        self._cache: dict[frozenset[str], CertifiedWorkloadValue] = {}

    def ca(self, acquired: Iterable[str], *, active_programs: Iterable[str] | None = None) -> CertifiedWorkloadValue:
        """Recheck certification; optionally exclude suspended/revoked programs."""
        known = frozenset(acquired)
        if not known.issubset(self.candidates):
            raise ValueError("unknown acquired evidence")
        if known not in self._cache or active_programs is not None:
            store = EvidenceStore(self.candidates[item] for item in sorted(known))
            programs = frozenset(self.certify_programs(store))
            if active_programs is not None:
                programs = programs.intersection(active_programs)
            unlocked = tuple(
                item for item in self.opportunities
                if any(support.issubset(known) for support in item.support_options)
                and item.required_programs.issubset(programs)
            )
            value = CertifiedWorkloadValue(
                sum(item.useful_value for item in unlocked),
                tuple(item.region_id for item in unlocked),
                sum(item.displaced_calls for item in unlocked),
            )
            if active_programs is not None:
                return value
            self._cache[known] = value
        return self._cache[known]

    def _expected_after(self, known: frozenset[str], bundle: frozenset[str]) -> tuple[float, float, tuple[str, ...], float, float]:
        before = self.ca(known)
        expected_delta = expected_calls = proof_cost = maintenance_cost = 0.0
        potentially_unlocked: set[str] = set()
        ordered = sorted(bundle)
        # Source-reviewed evidence normally has deterministic acceptance.  Do
        # not enumerate 2**|bundle| zero-probability outcomes in that common
        # case: large semantic evidence bundles are precisely where LPBA must
        # remain tractable.  This is algebraically identical to the loop below.
        if all(self.acceptance_probabilities.get(record_id, 1.0) == 1.0 for record_id in ordered):
            after = self.ca(known.union(bundle))
            new_regions = set(after.regions).difference(before.regions)
            attempted = {
                item.region_id for item in self.opportunities
                if item.region_id not in before.regions
                and any(support.issubset(known.union(bundle)) for support in item.support_options)
            }
            return (
                after.ca - before.ca,
                after.expected_model_calls_displaced - before.expected_model_calls_displaced,
                tuple(sorted(new_regions)),
                sum(item.proof_cost for item in self.opportunities if item.region_id in attempted),
                sum(item.maintenance_cost for item in self.opportunities if item.region_id in new_regions),
            )
        for accepted in product((False, True), repeat=len(ordered)):
            probability = 1.0
            answer_ids: set[str] = set()
            for record_id, success in zip(ordered, accepted, strict=True):
                chance = self.acceptance_probabilities.get(record_id, 1.0)
                probability *= chance if success else 1 - chance
                if success:
                    answer_ids.add(record_id)
            if probability == 0:
                continue
            after = self.ca(known.union(answer_ids))
            new_regions = set(after.regions).difference(before.regions)
            attempted = {
                item.region_id for item in self.opportunities
                if item.region_id not in before.regions
                and any(support.issubset(known.union(answer_ids)) for support in item.support_options)
            }
            expected_delta += probability * (after.ca - before.ca)
            expected_calls += probability * (after.expected_model_calls_displaced - before.expected_model_calls_displaced)
            proof_cost += probability * sum(item.proof_cost for item in self.opportunities if item.region_id in attempted)
            maintenance_cost += probability * sum(item.maintenance_cost for item in self.opportunities if item.region_id in new_regions)
            potentially_unlocked.update(new_regions)
        return expected_delta, expected_calls, tuple(sorted(potentially_unlocked)), proof_cost, maintenance_cost

    def choices(
        self,
        acquired: Iterable[str],
        *,
        budget: float,
        required_regions: Iterable[str] | None = None,
        total_cost_budget: float | None = None,
    ) -> tuple[BundleChoice, ...]:
        """Choose a complete missing support or a shared-support union.

        ``budget`` caps evidence purchases. ``total_cost_budget`` optionally
        caps the combined evidence, proof, and maintenance cost.  The method
        exposes every positive checked bundle so a model-independent caller can
        record online choice density before selecting the highest-ranked one.
        """
        if budget < 0 or (total_cost_budget is not None and total_cost_budget < 0):
            raise ValueError("acquisition budgets must be nonnegative")
        known = frozenset(acquired)
        already = set(self.ca(known).regions)
        required = frozenset(required_regions) if required_regions is not None else None
        if required is not None and not required.issubset({item.region_id for item in self.opportunities}):
            raise ValueError("unknown required workload region")
        missing = {
            support.difference(known)
            for item in self.opportunities if item.region_id not in already
            and (required is None or item.region_id in required)
            for support in item.support_options
        }
        missing.discard(frozenset())
        supports = tuple(missing)
        missing.update(left.union(right) for index, left in enumerate(supports) for right in supports[index + 1:])
        choices: list[BundleChoice] = []
        for bundle in missing:
            if not bundle:
                continue
            evidence_cost = sum(self.candidates[item].cost for item in bundle)
            if evidence_cost > budget:
                continue
            delta, calls, regions, proof_cost, maintenance_cost = self._expected_after(known, bundle)
            if required is not None and not required.intersection(regions):
                continue
            denominator = evidence_cost + proof_cost + maintenance_cost
            if total_cost_budget is not None and denominator > total_cost_budget:
                continue
            if delta <= 0 or denominator <= 0:
                continue
            choices.append(BundleChoice(tuple(sorted(bundle)), regions, delta, calls, evidence_cost, proof_cost, maintenance_cost, delta / denominator))
        return tuple(sorted(choices, key=lambda item: (-item.score, -item.expected_delta_ca, item.evidence_cost, item.evidence_ids)))

    def choose(
        self,
        acquired: Iterable[str],
        *,
        budget: float,
        required_regions: Iterable[str] | None = None,
        total_cost_budget: float | None = None,
    ) -> BundleChoice | None:
        choices = self.choices(
            acquired,
            budget=budget,
            required_regions=required_regions,
            total_cost_budget=total_cost_budget,
        )
        return choices[0] if choices else None

    def acquire_or_defer(
        self,
        acquired: Iterable[str],
        *,
        pending: Iterable[str] = (),
        current_region: str,
        budget: float,
        total_cost_budget: float | None,
        remaining_arrivals: int,
        delay_slots: int,
        minimum_proactive_score: float = 0.0,
        minimum_proactive_displaced_calls: float = 0.0,
    ) -> AcquireOrDeferDecision:
        """Select a bundle or explicitly wait using only current state.

        Pending evidence is treated as reserved for candidate construction,
        but not as trusted by the caller's runtime.  This prevents duplicate
        review requests while preserving the distinction between *scheduled*
        and *available* authority.  A current-region bundle may be acquired
        when it can help now (zero delay) or during the remaining stream. A
        non-current bundle requires either shared/complementary authority or
        enough expected displaced fallback calls to justify acting early.
        """
        if remaining_arrivals < 0 or delay_slots < 0:
            raise ValueError("remaining arrivals and delay slots must be nonnegative")
        if minimum_proactive_score < 0 or minimum_proactive_displaced_calls < 0:
            raise ValueError("acquire-or-defer thresholds must be nonnegative")
        trusted = frozenset(acquired)
        reserved = frozenset(pending)
        if not trusted.isdisjoint(reserved):
            reserved = reserved.difference(trusted)
        planned = trusted.union(reserved)
        if current_region not in {item.region_id for item in self.opportunities}:
            return AcquireOrDeferDecision("defer", None, 0.0, 0.0, 0.0, "unknown_current_region")
        ready_fraction = max(0, remaining_arrivals - delay_slots) / max(1, remaining_arrivals)
        current = next(item for item in self.opportunities if item.region_id == current_region)
        choices = self.choices(
            planned,
            budget=budget,
            required_regions=None,
            total_cost_budget=total_cost_budget,
        )
        ranked: list[tuple[float, BundleChoice, float, float, str]] = []
        for choice in choices:
            ready_ca = choice.expected_delta_ca * ready_fraction
            ready_calls = choice.expected_model_calls_displaced * ready_fraction
            serves_current = current_region in choice.unlocked_regions
            immediate_ca = current.useful_value_per_occurrence if serves_current and delay_slots == 0 else 0.0
            denominator = choice.evidence_cost + choice.proof_cost + choice.maintenance_cost
            effective_score = (ready_ca + immediate_ca) / denominator if denominator else 0.0
            if serves_current and (immediate_ca > 0 or ready_ca > 0):
                ranked.append((effective_score, choice, ready_ca, ready_calls, "current_demand"))
                continue
            available = planned.union(choice.evidence_ids)
            shared_authority = any(
                sum(
                    any(record_id in support and support.issubset(available) for support in item.support_options)
                    for item in self.opportunities if item.region_id in choice.unlocked_regions
                ) > 1
                for record_id in choice.evidence_ids
            )
            if (
                ready_ca > 0
                and effective_score >= minimum_proactive_score
                and (shared_authority or ready_calls > minimum_proactive_displaced_calls)
            ):
                ranked.append((effective_score, choice, ready_ca, ready_calls, "proactive_shared_or_fallback_value"))
        if not ranked:
            return AcquireOrDeferDecision("defer", None, 0.0, 0.0, 0.0, "waiting_has_at_least_as_much_declared_value")
        # Current demand breaks ties so zero-delay separable workloads reduce
        # to First-Use rather than arbitrary speculative pre-purchase.
        ranked.sort(
            key=lambda item: (
                -item[0],
                -(current_region in item[1].unlocked_regions),
                -item[1].expected_delta_ca,
                item[1].evidence_cost,
                item[1].evidence_ids,
            )
        )
        score, choice, ready_ca, ready_calls, reason = ranked[0]
        return AcquireOrDeferDecision("acquire", choice, score, ready_ca, ready_calls, reason)
