"""Conservative, model-independent routing for proof-economic delegation.

This module deliberately sits *above* evidence review, certification, and the
deterministic runtime.  It never admits evidence, accepts a proof, or invokes
an LLM.  Instead it selects the acquisition posture that the existing checked
components should execute.  This separation lets the same routing logic work
with Qwen, a mock adapter, or another model adapter.

The router is intentionally conservative.  A point estimate that a speculative
bundle might be useful is insufficient: proactive or eager formalization needs
a lower bound showing an advantage over First-Use without a worse upper bound
on review/proof or maintenance cost.  In the absence of that evidence it
returns the demand-driven First-Use baseline (or DEFER when there is no current
eligible region).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from itertools import combinations
from typing import Iterable, Mapping

from .evidence import EvidenceRecord
from .skill_acquire import BundleChoice, SkillBundlePlanner, SkillOpportunity


class TimingPolicy(str, Enum):
    """Level 1: when authority should be acquired."""

    PREDEPLOY = "predeploy"
    PROACTIVE = "proactive"
    FIRST_USE = "first_use"
    DEFER = "defer"

    # Kept as Enum aliases for callers of the earlier one-level controller.
    # New results always serialize the canonical two-level names above.
    EAGER_FORMALIZATION = PREDEPLOY
    PROACTIVE_BUNDLE_ACQUISITION = PROACTIVE
    LAZY_FIRST_USE = FIRST_USE


# Compatibility alias; new code should use TimingPolicy.
ProofEconomicStrategy = TimingPolicy


class EvidenceSelectionPolicy(str, Enum):
    """Level 2: how a permitted acquisition identifies evidence."""

    NONE = "none"
    FULL_CATALOG = "full_catalog"
    DIRECT_COMPLETE_SUPPORT = "direct_complete_support"
    LPBA_BUNDLE = "lpba_bundle"


class ControllerObjective(str, Enum):
    """The declared policy objective; never a hidden scalar conversion."""

    BUDGETED_AUTHORITY = "budgeted_authority"
    ECONOMIC_EFFICIENCY = "economic_efficiency"


@dataclass(frozen=True, slots=True)
class LifecycleBenefitLowerBound:
    """A unit-preserving conservative comparison with First-Use.

    Useful workload, fallback calls, and authority downtime are kept separate;
    the controller does not invent a review-unit-to-token or currency
    conversion.  A proactive action is justified only when it has a strictly
    positive lower bound in at least one benefit coordinate *and* non-positive
    upper bounds for its extra review/proof and maintenance cost.

    The values must be constructed from a frozen development prior and the
    observed prefix.  They are inputs to routing rather than measurements from
    future held-out arrivals.
    """

    useful_workload_gain_lower_bound: float = 0.0
    fallback_calls_avoided_lower_bound: float = 0.0
    authority_downtime_avoided_lower_bound: float = 0.0
    additional_review_proof_cost_upper_bound: float = 0.0
    additional_maintenance_cost_upper_bound: float = 0.0
    basis: str = "no certified advantage supplied"

    def __post_init__(self) -> None:
        if any(value < 0 for value in (
            self.useful_workload_gain_lower_bound,
            self.fallback_calls_avoided_lower_bound,
            self.authority_downtime_avoided_lower_bound,
        )):
            raise ValueError("benefit lower bounds must be nonnegative")

    @property
    def is_positive_without_extra_lifecycle_cost(self) -> bool:
        return (
            self.additional_review_proof_cost_upper_bound <= 0
            and self.additional_maintenance_cost_upper_bound <= 0
            and (
                self.useful_workload_gain_lower_bound > 0
                or self.fallback_calls_avoided_lower_bound > 0
                or self.authority_downtime_avoided_lower_bound > 0
            )
        )


@dataclass(frozen=True, slots=True)
class StructuralWorkloadFeatures:
    """Observable proof/workload structure used by the routing rule.

    Fields are portfolio-level quantities.  ``specification_headroom_estimate``
    and the benefit bounds are allowed-information estimates; they must never
    be filled from future held-out family occurrences.  ``None`` represents an
    unavailable feature rather than an assumed favourable value.
    """

    total_candidate_evidence_cost: float
    total_candidate_proof_cost: float
    specification_headroom_estimate: float | None
    proof_support_overlap: float
    conjunctive_support_fraction: float
    acquisition_choice_density: float
    allowed_information_demand_predictability: float | None
    evidence_acquisition_delay_slots: int
    acquisition_budget: float
    source_volatility: float | None
    proactive_lifecycle_benefit: LifecycleBenefitLowerBound = LifecycleBenefitLowerBound()
    predeployment_lifecycle_benefit: LifecycleBenefitLowerBound = LifecycleBenefitLowerBound()
    predeployment_window: bool = False
    feature_basis: str = "development catalog plus observed workload prefix"

    def __post_init__(self) -> None:
        if any(value < 0 for value in (
            self.total_candidate_evidence_cost, self.total_candidate_proof_cost, self.acquisition_budget,
        )):
            raise ValueError("catalog cost and acquisition budget must be nonnegative")
        if self.evidence_acquisition_delay_slots < 0:
            raise ValueError("evidence delay must be nonnegative")
        for name, value in (
            ("specification_headroom_estimate", self.specification_headroom_estimate),
            ("proof_support_overlap", self.proof_support_overlap),
            ("conjunctive_support_fraction", self.conjunctive_support_fraction),
            ("acquisition_choice_density", self.acquisition_choice_density),
            ("allowed_information_demand_predictability", self.allowed_information_demand_predictability),
            ("source_volatility", self.source_volatility),
        ):
            if value is not None and not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1] when available")

    @property
    def budget_catalog_ratio(self) -> float | None:
        complete_cost = self.total_candidate_evidence_cost + self.total_candidate_proof_cost
        if complete_cost == 0:
            return None
        return self.acquisition_budget / complete_cost

    @property
    def evidence_budget_catalog_ratio(self) -> float | None:
        if self.total_candidate_evidence_cost == 0:
            return None
        return self.acquisition_budget / self.total_candidate_evidence_cost

    @property
    def has_complementary_or_competing_structure(self) -> bool:
        return (
            self.proof_support_overlap > 0
            or self.conjunctive_support_fraction > 0
            or self.acquisition_choice_density > 0
        )

    def as_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["budget_catalog_ratio"] = self.budget_catalog_ratio
        result["evidence_budget_catalog_ratio"] = self.evidence_budget_catalog_ratio
        result["has_complementary_or_competing_structure"] = self.has_complementary_or_competing_structure
        return result


def structural_features_from_catalog(
    candidates: Iterable[EvidenceRecord],
    opportunities: Iterable[SkillOpportunity],
    *,
    acquisition_budget: float,
    evidence_acquisition_delay_slots: int,
    specification_headroom_estimate: float | None = None,
    acquisition_choice_density: float = 0.0,
    allowed_information_demand_predictability: float | None = None,
    source_volatility: float | None = None,
    proactive_lifecycle_benefit: LifecycleBenefitLowerBound | None = None,
    predeployment_lifecycle_benefit: LifecycleBenefitLowerBound | None = None,
    predeployment_window: bool = False,
    feature_basis: str = "development catalog plus observed workload prefix",
) -> StructuralWorkloadFeatures:
    """Compute catalog structure without inspecting future workload arrivals.

    For regions with alternative proofs, the least-cost support is used for
    the overlap summary.  This is deterministic, source-derived, and avoids
    calling an LLM or a test-set outcome oracle.
    """

    records = tuple(candidates)
    by_id: Mapping[str, EvidenceRecord] = {record.id: record for record in records}
    if len(by_id) != len(records):
        raise ValueError("candidate evidence IDs must be unique")
    regions = tuple(opportunities)
    supports: list[frozenset[str]] = []
    conjunctive = 0
    total_options = 0
    for opportunity in regions:
        if any(not option.issubset(by_id) for option in opportunity.support_options):
            raise ValueError("opportunity support lies outside candidate catalog")
        ordered = sorted(
            opportunity.support_options,
            key=lambda option: (sum(by_id[item].cost for item in option), len(option), tuple(sorted(option))),
        )
        supports.append(ordered[0])
        conjunctive += sum(len(option) > 1 for option in opportunity.support_options)
        total_options += len(opportunity.support_options)
    overlap_values = [
        len(left.intersection(right)) / len(left.union(right))
        for left, right in combinations(supports, 2)
        if left.union(right)
    ]
    return StructuralWorkloadFeatures(
        total_candidate_evidence_cost=sum(record.cost for record in records),
        total_candidate_proof_cost=sum(item.proof_cost for item in regions),
        specification_headroom_estimate=specification_headroom_estimate,
        proof_support_overlap=sum(overlap_values) / len(overlap_values) if overlap_values else 0.0,
        conjunctive_support_fraction=conjunctive / total_options if total_options else 0.0,
        acquisition_choice_density=acquisition_choice_density,
        allowed_information_demand_predictability=allowed_information_demand_predictability,
        evidence_acquisition_delay_slots=evidence_acquisition_delay_slots,
        acquisition_budget=acquisition_budget,
        source_volatility=source_volatility,
        proactive_lifecycle_benefit=proactive_lifecycle_benefit or LifecycleBenefitLowerBound(),
        predeployment_lifecycle_benefit=predeployment_lifecycle_benefit or LifecycleBenefitLowerBound(),
        predeployment_window=predeployment_window,
        feature_basis=feature_basis,
    )


@dataclass(frozen=True, slots=True)
class ProofEconomicDecision:
    """A two-level selection and, when applicable, a checked candidate bundle."""

    timing_policy: TimingPolicy
    evidence_selection_policy: EvidenceSelectionPolicy
    bundle: BundleChoice | None
    evidence_ids: tuple[str, ...]
    use_existing_authority: bool
    reason: str
    features: StructuralWorkloadFeatures

    @property
    def acquires_evidence(self) -> bool:
        return bool(self.evidence_ids)

    @property
    def strategy(self) -> TimingPolicy:
        """Compatibility view for the earlier single-level controller API."""

        return self.timing_policy

    def as_dict(self) -> dict[str, object]:
        return {
            "timing_policy": self.timing_policy.value,
            "evidence_selection_policy": self.evidence_selection_policy.value,
            "strategy": self.timing_policy.value,
            "bundle": asdict(self.bundle) if self.bundle is not None else None,
            "evidence_ids": list(self.evidence_ids),
            "use_existing_authority": self.use_existing_authority,
            "reason": self.reason,
            "features": self.features.as_dict(),
        }


class ProofEconomicDelegationController:
    """Route acquisition without changing the trusted delegation machinery.

    The controller makes only a scheduling decision.  Callers still review
    evidence, run the existing certifier, register certificates, and execute
    through the existing guarded runtime.  ``SkillBundlePlanner`` is used only
    to enumerate checked candidate bundles after routing has permitted an
    acquisition posture.
    """

    def select_timing_policy(
        self,
        features: StructuralWorkloadFeatures,
        *,
        current_region_available: bool,
        existing_authority_available: bool = False,
        pending_authority_available: bool = False,
        objective: ControllerObjective = ControllerObjective.ECONOMIC_EFFICIENCY,
    ) -> tuple[TimingPolicy, str]:
        """Level 1: select *when* to acquire using legal inputs only.

        Full predeployment is deliberately valid under the hard-cap authority
        objective when its full catalog fits and delay is nonzero.  It is not
        thereby declared economically Pareto-optimal; the caller must report
        the cost/useful-workload/fallback frontier separately.
        """

        if existing_authority_available:
            return TimingPolicy.DEFER, "checked authority is already active; acquisition is unnecessary"
        if pending_authority_available:
            return TimingPolicy.DEFER, "required authority is already pending review; avoid duplicate acquisition"

        catalog_fits = features.budget_catalog_ratio is None or features.budget_catalog_ratio >= 1.0
        if (
            features.predeployment_window
            and features.evidence_acquisition_delay_slots > 0
            and catalog_fits
            and (
                objective is ControllerObjective.BUDGETED_AUTHORITY
                or features.predeployment_lifecycle_benefit.is_positive_without_extra_lifecycle_cost
            )
        ):
            return TimingPolicy.PREDEPLOY, "complete catalog fits the declared hard budget during a nonzero-delay predeployment window"

        if not current_region_available:
            return TimingPolicy.DEFER, "no current eligible workload region; preserve the portfolio and fall back"
        if features.evidence_acquisition_delay_slots == 0:
            return TimingPolicy.FIRST_USE, "zero delay permits same-request checked First-Use"
        if not features.proactive_lifecycle_benefit.is_positive_without_extra_lifecycle_cost:
            return TimingPolicy.FIRST_USE, "no positive conservative proactive lifecycle benefit is established"
        return TimingPolicy.PROACTIVE, "conservative allowed-information lifecycle benefit over First-Use is positive"

    def select_strategy(
        self,
        features: StructuralWorkloadFeatures,
        *,
        current_region_available: bool,
        existing_authority_available: bool = False,
    ) -> tuple[TimingPolicy, str]:
        """Backward-compatible economic-efficiency timing entry point."""

        return self.select_timing_policy(
            features,
            current_region_available=current_region_available,
            existing_authority_available=existing_authority_available,
        )

    @staticmethod
    def _current_opportunity(planner: SkillBundlePlanner, region_id: str | None) -> SkillOpportunity | None:
        return next((item for item in planner.opportunities if item.region_id == region_id), None)

    def select_evidence_policy(
        self,
        planner: SkillBundlePlanner,
        *,
        acquired: Iterable[str],
        current_region: str | None,
        timing_policy: TimingPolicy,
    ) -> EvidenceSelectionPolicy:
        """Level 2: choose direct support versus bundle-aware LPBA.

        A single predeclared support for the currently demanded region is
        directly acquired.  Bundle-aware LPBA is reserved for alternative,
        shared/overlapping, conjunctive, or answer-dependent proof choices.
        """

        if timing_policy is TimingPolicy.DEFER:
            return EvidenceSelectionPolicy.NONE
        if timing_policy is TimingPolicy.PREDEPLOY:
            return EvidenceSelectionPolicy.FULL_CATALOG
        current = self._current_opportunity(planner, current_region)
        if timing_policy is TimingPolicy.FIRST_USE and current is not None and len(current.support_options) == 1:
            support = next(iter(current.support_options))
            if all(planner.acceptance_probabilities.get(record_id, 1.0) == 1.0 for record_id in support):
                return EvidenceSelectionPolicy.DIRECT_COMPLETE_SUPPORT
        if any(len(item.support_options) > 1 for item in planner.opportunities):
            return EvidenceSelectionPolicy.LPBA_BUNDLE
        unresolved = [
            support.difference(frozenset(acquired))
            for item in planner.opportunities
            for support in item.support_options
        ]
        shared_support = any(left and right and left.intersection(right) for left, right in combinations(unresolved, 2))
        answer_dependent = any(value < 1.0 for value in planner.acceptance_probabilities.values())
        if shared_support or answer_dependent or any(len(support) > 1 for support in unresolved):
            return EvidenceSelectionPolicy.LPBA_BUNDLE
        return EvidenceSelectionPolicy.DIRECT_COMPLETE_SUPPORT

    def decide(
        self,
        planner: SkillBundlePlanner,
        *,
        acquired: Iterable[str],
        current_region: str | None,
        evidence_budget: float,
        total_cost_budget: float | None,
        features: StructuralWorkloadFeatures,
        existing_authority_available: bool = False,
        pending_authority_available: bool = False,
        objective: ControllerObjective = ControllerObjective.ECONOMIC_EFFICIENCY,
    ) -> ProofEconomicDecision:
        """Select a posture and an evidence request, never trusted evidence.

        ``evidence_budget`` and ``total_cost_budget`` are the caller's
        remaining portfolio budgets.  The return value is a request; only the
        existing review/certification path may turn it into active authority.
        """

        if evidence_budget < 0 or (total_cost_budget is not None and total_cost_budget < 0):
            raise ValueError("remaining acquisition budgets must be nonnegative")
        region_ids = {item.region_id for item in planner.opportunities}
        current_valid = current_region is not None and current_region in region_ids
        timing_policy, reason = self.select_timing_policy(
            features,
            current_region_available=current_valid,
            existing_authority_available=existing_authority_available,
            pending_authority_available=pending_authority_available,
            objective=objective,
        )
        known = frozenset(acquired)
        evidence_selection_policy = self.select_evidence_policy(
            planner, acquired=known, current_region=current_region, timing_policy=timing_policy,
        )
        if timing_policy is TimingPolicy.DEFER:
            return ProofEconomicDecision(
                timing_policy, evidence_selection_policy, None, (), existing_authority_available, reason, features,
            )

        if timing_policy is TimingPolicy.PREDEPLOY:
            evidence_ids = tuple(sorted(set(planner.candidates).difference(known)))
            evidence_cost = sum(planner.candidates[item].cost for item in evidence_ids)
            after = planner.ca(known.union(evidence_ids))
            before = planner.ca(known)
            new_regions = set(after.regions).difference(before.regions)
            proof_cost = sum(item.proof_cost for item in planner.opportunities if item.region_id in new_regions)
            maintenance_cost = sum(item.maintenance_cost for item in planner.opportunities if item.region_id in new_regions)
            total_cost = evidence_cost + proof_cost + maintenance_cost
            if evidence_cost > evidence_budget or (total_cost_budget is not None and total_cost > total_cost_budget):
                return ProofEconomicDecision(
                    TimingPolicy.DEFER, EvidenceSelectionPolicy.NONE, None, (), False,
                    "predeployment route was selected but the remaining hard budget cannot fund the complete checked catalog", features,
                )
            return ProofEconomicDecision(
                timing_policy, evidence_selection_policy, None, evidence_ids, False, reason, features,
            )

        if timing_policy is TimingPolicy.FIRST_USE:
            choices = planner.choices(
                known,
                budget=evidence_budget,
                required_regions=(current_region,),
                total_cost_budget=total_cost_budget,
            )
            matching = [choice for choice in choices if current_region in choice.unlocked_regions]
            if not matching:
                return ProofEconomicDecision(
                    TimingPolicy.DEFER, EvidenceSelectionPolicy.NONE, None, (), False,
                    "First-Use has no affordable complete checked support; defer to model fallback", features,
                )
            choice = min(
                matching,
                key=lambda item: (
                    item.evidence_cost + item.proof_cost + item.maintenance_cost,
                    item.evidence_ids,
                ),
            )
            return ProofEconomicDecision(
                timing_policy, evidence_selection_policy, choice, choice.evidence_ids, False, reason, features,
            )

        if evidence_selection_policy is EvidenceSelectionPolicy.DIRECT_COMPLETE_SUPPORT:
            choices = planner.choices(
                known,
                budget=evidence_budget,
                required_regions=(current_region,) if current_region is not None else None,
                total_cost_budget=total_cost_budget,
            )
        else:
            choices = planner.choices(known, budget=evidence_budget, total_cost_budget=total_cost_budget)
        if not choices:
            return ProofEconomicDecision(
                TimingPolicy.DEFER, EvidenceSelectionPolicy.NONE, None, (), False,
                "proactive route has no affordable positive checked bundle; defer to model fallback", features,
            )
        choice = choices[0]
        return ProofEconomicDecision(
            timing_policy, evidence_selection_policy, choice, choice.evidence_ids, False, reason, features,
        )
