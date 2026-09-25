"""Exact minimal-support and acquisition-order oracle for small Horn theories.

This is an offline formal-source analysis. It does not infer source semantics
from natural language, inspect benchmark answers, or authorize agent actions.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import inf
from typing import Iterable, Mapping, Sequence

import numpy as np

from ..acquire import WorkItem
from ..evidence import EvidenceKind, EvidenceRecord
from ..ir import HornRule, Literal


class SupportLimitError(RuntimeError):
    """Exact enumeration exceeded its predeclared resource limit."""


def _insert_minimal(masks: set[int], candidate: int, limit: int) -> bool:
    if any(existing & candidate == existing for existing in masks):
        return False
    masks.difference_update([existing for existing in masks if existing & candidate == candidate])
    masks.add(candidate)
    if len(masks) > limit:
        raise SupportLimitError(f"minimal-support antichain exceeded {limit}")
    return True


def enumerate_minimal_supports(
    candidates: Sequence[EvidenceRecord],
    workload: Sequence[WorkItem],
    *,
    limit: int,
) -> dict[str, tuple[int, ...]]:
    """Compute all inclusion-minimal supports under finite positive Horn logic.

    Bit i denotes the ith candidate in sorted-id order. At every literal and
    partial rule join, a superset proof is discarded because it cannot improve
    certification under any acquired set. Closure stops only at a fixed point.
    """
    records = sorted(candidates, key=lambda record: record.id)
    if len({record.id for record in records}) != len(records):
        raise ValueError("candidate evidence ids must be unique")
    known: dict[Literal, set[int]] = defaultdict(set)
    rules: list[tuple[int, HornRule]] = []
    for index, record in enumerate(records):
        bit = 1 << index
        if record.kind is EvidenceKind.FACT:
            if not isinstance(record.payload, Literal) or not record.payload.is_ground():
                raise ValueError("exact support enumerator requires ground facts")
            _insert_minimal(known[record.payload], bit, limit)
        elif record.kind is EvidenceKind.RULE:
            if not isinstance(record.payload, HornRule) or not record.payload.is_safe():
                raise ValueError("exact support enumerator requires safe Horn rules")
            rules.append((bit, record.payload))
        else:
            raise ValueError(f"unsupported formal evidence kind: {record.kind}")

    # Every improving iteration introduces at least one new minimal support.
    # The resource limit is explicit; a cap is never silently treated as exact.
    for _round in range(256):
        changed = False
        snapshot = {literal: tuple(sorted(masks)) for literal, masks in known.items()}
        for rule_bit, rule in rules:
            partials: dict[tuple[tuple[str, str], ...], set[int]] = {(): {rule_bit}}
            for atom in rule.body:
                next_partials: dict[tuple[tuple[str, str], ...], set[int]] = defaultdict(set)
                compatible = [
                    (literal, masks)
                    for literal, masks in snapshot.items()
                    if literal.predicate == atom.predicate and literal.positive == atom.positive
                ]
                for binding_tuple, partial_masks in partials.items():
                    bindings = dict(binding_tuple)
                    for literal, premise_masks in compatible:
                        matched = atom.matches(literal, bindings)
                        if matched is None:
                            continue
                        target = next_partials[tuple(sorted(matched.items()))]
                        for left in partial_masks:
                            for right in premise_masks:
                                _insert_minimal(target, left | right, limit)
                partials = next_partials
                if not partials:
                    break
            for binding_tuple, masks in partials.items():
                head = rule.head.substitute(dict(binding_tuple))
                if not head.is_ground():
                    continue
                target = known[head]
                for mask in masks:
                    changed |= _insert_minimal(target, mask, limit)
        if not changed:
            break
    else:
        raise SupportLimitError("minimal-support closure did not converge in 256 rounds")

    result: dict[str, tuple[int, ...]] = {}
    for item in workload:
        supports: set[int] = set()
        for mask in known.get(item.target, ()):
            _insert_minimal(supports, mask, limit)
        for mask in known.get(item.target.negate(), ()):
            _insert_minimal(supports, mask, limit)
        result[item.item_id] = tuple(sorted(supports, key=lambda mask: (mask.bit_count(), mask)))
    return result


@dataclass(slots=True)
class Catalog:
    ids: tuple[str, ...]
    costs: tuple[float, ...]
    workload: tuple[WorkItem, ...]
    supports: dict[str, tuple[int, ...]]

    @classmethod
    def build(
        cls,
        candidates: Sequence[EvidenceRecord],
        workload: Sequence[WorkItem],
        *,
        support_limit: int,
    ) -> "Catalog":
        records = sorted(candidates, key=lambda record: record.id)
        return cls(
            tuple(record.id for record in records),
            tuple(float(record.cost) for record in records),
            tuple(workload),
            enumerate_minimal_supports(records, workload, limit=support_limit),
        )

    @property
    def relevant_mask(self) -> int:
        mask = 0
        for supports in self.supports.values():
            for support in supports:
                mask |= support
        return mask

    @property
    def relevant_count(self) -> int:
        return self.relevant_mask.bit_count()

    def value(self, mask: int) -> float:
        return sum(
            item.weight
            for item in self.workload
            if any(mask & support == support for support in self.supports[item.item_id])
        )

    def support_count(self) -> int:
        return sum(len(supports) for supports in self.supports.values())

    def feature_values(self) -> dict[str, float]:
        certifiable = [item for item in self.workload if self.supports[item.item_id]]
        total_weight = sum(item.weight for item in certifiable)
        complementary_weight = sum(
            item.weight
            for item in certifiable
            if min(mask.bit_count() for mask in self.supports[item.item_id]) >= 2
        )
        appearances = [0] * len(self.ids)
        for item in self.workload:
            item_mask = 0
            for support in self.supports[item.item_id]:
                item_mask |= support
            for index in range(len(self.ids)):
                appearances[index] += bool(item_mask & (1 << index))
        relevant_cost = sum(cost for cost, count in zip(self.costs, appearances) if count)
        shared_cost = sum(cost for cost, count in zip(self.costs, appearances) if count >= 2)
        weights = [item.weight for item in self.workload]
        mean_weight = sum(weights) / len(weights) if weights else 0.0
        variance = sum((weight - mean_weight) ** 2 for weight in weights) / len(weights) if weights else 0.0
        return {
            "complementarity_fraction": complementary_weight / total_weight if total_weight else 0.0,
            "shared_evidence_cost_fraction": shared_cost / relevant_cost if relevant_cost else 0.0,
            "workload_weight_cv": variance**0.5 / mean_weight if mean_weight else 0.0,
        }


def _acquired_from_order(catalog: Catalog, order: Sequence[int]) -> Iterable[tuple[float, float]]:
    mask = 0
    for index in order:
        yield catalog.costs[index], catalog.value(mask)
        mask |= 1 << index


def feasible_auc(catalog: Catalog, order: Sequence[int]) -> float:
    if set(order) != set(range(len(catalog.ids))) or len(order) != len(catalog.ids):
        raise ValueError("policy order must purchase every source item exactly once")
    total_cost = sum(catalog.costs)
    if not total_cost:
        return 0.0
    return sum(cost * value for cost, value in _acquired_from_order(catalog, order)) / total_cost


def prefix_value(catalog: Catalog, order: Sequence[int], budget: float) -> float:
    spent = 0.0
    acquired = 0
    for index in order:
        cost = catalog.costs[index]
        if spent + cost > budget + 1e-9:
            break
        spent += cost
        acquired |= 1 << index
    return catalog.value(acquired)


def _finish_order(catalog: Catalog, order: list[int]) -> tuple[int, ...]:
    seen = set(order)
    order.extend(index for index in range(len(catalog.ids)) if index not in seen)
    return tuple(order)


def one_step_order(catalog: Catalog) -> tuple[int, ...]:
    order: list[int] = []
    acquired = 0
    remaining = set(range(len(catalog.ids)))
    while remaining:
        before = catalog.value(acquired)
        selected = min(
            remaining,
            key=lambda index: (
                -(catalog.value(acquired | (1 << index)) - before) / catalog.costs[index],
                catalog.ids[index],
            ),
        )
        order.append(selected)
        acquired |= 1 << selected
        remaining.remove(selected)
    return tuple(order)


def decision_directed_order(catalog: Catalog) -> tuple[int, ...]:
    order: list[int] = []
    acquired = 0
    remaining = set(range(len(catalog.ids)))
    while remaining:
        scores = {index: 0.0 for index in remaining}
        for item in catalog.workload:
            for support in catalog.supports[item.item_id]:
                missing = support & ~acquired
                if not missing:
                    continue
                contribution = item.weight / missing.bit_count()
                for index in remaining:
                    if missing & (1 << index):
                        scores[index] += contribution / catalog.costs[index]
        selected = min(remaining, key=lambda index: (-scores[index], catalog.ids[index]))
        order.append(selected)
        acquired |= 1 << selected
        remaining.remove(selected)
    return tuple(order)


def lazy_first_use_order(catalog: Catalog) -> tuple[int, ...]:
    order: list[int] = []
    acquired = 0
    for item in catalog.workload:
        supports = catalog.supports[item.item_id]
        if not supports or any(acquired & support == support for support in supports):
            continue
        chosen = min(
            supports,
            key=lambda support: (
                sum(cost for index, cost in enumerate(catalog.costs) if support & ~acquired & (1 << index)),
                (support & ~acquired).bit_count(),
                tuple(catalog.ids[index] for index in range(len(catalog.ids)) if support & ~acquired & (1 << index)),
            ),
        )
        for index in range(len(catalog.ids)):
            if chosen & ~acquired & (1 << index):
                order.append(index)
                acquired |= 1 << index
    return _finish_order(catalog, order)


def lpba_order(catalog: Catalog, *, max_local_union: int = 3, max_bundle_candidates: int = 32) -> tuple[int, ...]:
    """The repository LPBA criterion on the complete minimal-support catalog."""
    order: list[int] = []
    acquired = 0
    remaining = set(range(len(catalog.ids)))
    while remaining:
        already = {
            item.item_id
            for item in catalog.workload
            if any(acquired & support == support for support in catalog.supports[item.item_id])
        }
        unfinished: list[tuple[WorkItem, int]] = []
        bundles: set[int] = set()
        for item in catalog.workload:
            for support in catalog.supports[item.item_id]:
                missing = support & ~acquired
                if missing:
                    unfinished.append((item, missing))
                    bundles.add(missing)
        for position, (_, left) in enumerate(unfinished):
            for _, right in unfinished[position + 1 :]:
                union = left | right
                if union.bit_count() <= max_local_union:
                    bundles.add(union)
                if len(bundles) >= max_bundle_candidates * 4:
                    break
            if len(bundles) >= max_bundle_candidates * 4:
                break
        if not bundles:
            # No certifiable work remains. All further acquisitions have zero
            # marginal value, so their order is immaterial to feasible AUC.
            return _finish_order(catalog, order)
        ranked: list[tuple[float, int, tuple[str, ...], int]] = []
        for bundle in bundles:
            cost = sum(catalog.costs[index] for index in remaining if bundle & (1 << index))
            if cost <= 0:
                continue
            value = sum(
                item.weight
                for item in catalog.workload
                if item.item_id not in already
                and any((support & ~acquired) & bundle == (support & ~acquired) for support in catalog.supports[item.item_id])
            )
            names = tuple(catalog.ids[index] for index in range(len(catalog.ids)) if bundle & (1 << index))
            ranked.append((-value / cost, bundle.bit_count(), names, bundle))
        if not ranked:
            return _finish_order(catalog, order)
        bundle = min(ranked)[3]
        influence = {
            index: sum(1 for _, missing in unfinished if missing & (1 << index)) / catalog.costs[index]
            for index in remaining
            if bundle & (1 << index)
        }
        selected = min(influence, key=lambda index: (-influence[index], catalog.ids[index]))
        order.append(selected)
        acquired |= 1 << selected
        remaining.remove(selected)
    return tuple(order)


@dataclass(slots=True)
class OracleResult:
    auc: float
    order: tuple[int, ...]
    budget_frontier: dict[float, float]


def exact_oracle(catalog: Catalog, budget_fractions: Sequence[float], *, max_relevant: int) -> OracleResult:
    """Exact subset DP for AUC and exhaustive finite-budget coverage frontier."""
    relevant_indices = tuple(index for index in range(len(catalog.ids)) if catalog.relevant_mask & (1 << index))
    if len(relevant_indices) > max_relevant:
        raise SupportLimitError(f"{len(relevant_indices)} relevant evidence items exceed limit {max_relevant}")
    full_to_local = {global_index: local_index for local_index, global_index in enumerate(relevant_indices)}
    local_supports: dict[str, tuple[int, ...]] = {}
    for item in catalog.workload:
        converted = []
        for support in catalog.supports[item.item_id]:
            mask = 0
            for global_index, local_index in full_to_local.items():
                if support & (1 << global_index):
                    mask |= 1 << local_index
            converted.append(mask)
        local_supports[item.item_id] = tuple(converted)
    states = 1 << len(relevant_indices)
    masks = np.arange(states, dtype=np.uint32)
    values = np.zeros(states, dtype=np.float64)
    for item in catalog.workload:
        covered = np.zeros(states, dtype=np.bool_)
        for support in local_supports[item.item_id]:
            covered |= (masks & support) == support
        values += item.weight * covered
    local_costs = tuple(catalog.costs[index] for index in relevant_indices)
    subset_costs = np.zeros(states, dtype=np.float64)
    for local_index, cost in enumerate(local_costs):
        subset_costs += cost * ((masks >> local_index) & 1)
    total_cost = sum(catalog.costs)
    frontier = {
        float(fraction): float(np.max(values[subset_costs <= total_cost * fraction + 1e-9]))
        for fraction in budget_fractions
    }
    if not relevant_indices:
        return OracleResult(0.0, tuple(range(len(catalog.ids))), frontier)
    dp = np.full(states, -inf, dtype=np.float64)
    parent = np.full(states, -1, dtype=np.int16)
    dp[0] = 0.0
    for mask in range(states):
        base = float(dp[mask])
        available = (states - 1) ^ mask
        while available:
            bit = available & -available
            local_index = bit.bit_length() - 1
            next_mask = mask | bit
            score = base + local_costs[local_index] * values[mask]
            if score > dp[next_mask] + 1e-12:
                dp[next_mask] = score
                parent[next_mask] = local_index
            available ^= bit
    reverse_order: list[int] = []
    current = states - 1
    while current:
        local_index = int(parent[current])
        if local_index < 0:
            raise RuntimeError("exact acquisition DP has no parent for a reachable subset")
        reverse_order.append(relevant_indices[local_index])
        current ^= 1 << local_index
    order = list(reversed(reverse_order))
    relevant_set = set(relevant_indices)
    order.extend(index for index in range(len(catalog.ids)) if index not in relevant_set)
    irrelevant_cost = total_cost - sum(local_costs)
    auc = (float(dp[-1]) + irrelevant_cost * float(values[-1])) / total_cost if total_cost else 0.0
    return OracleResult(auc, tuple(order), frontier)
