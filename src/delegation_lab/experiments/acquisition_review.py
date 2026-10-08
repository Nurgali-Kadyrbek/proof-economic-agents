"""Independent latency audit and complete-bundle density controls.

The exact-density comparator uses the maximum incremental value/cost over
all remaining subsets. This is the greedy density primitive in min-sum
ordering, not a hidden-outcome HEC implementation. A chosen block is completed
before replanning; inside it, source participation/cost determines order.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .proofwriter_exact_core import Catalog, _finish_order


def weighted_completion_cost(catalog: Catalog, order: Sequence[int]) -> float:
    """Sum w_j times first complete-support cost, excluding unprovable items.

    Unlike relative area regret, this objective is invariant to irrelevant
    sources appended after the last certification.
    """
    if sorted(order) != list(range(len(catalog.ids))):
        raise ValueError("order must contain each source once")
    acquired = 0
    spent = 0.0
    done: set[str] = set()
    answer = 0.0
    for index in order:
        spent += catalog.costs[index]
        acquired |= 1 << index
        for item in catalog.workload:
            if item.item_id not in done and any(acquired & p == p for p in catalog.supports[item.item_id]):
                done.add(item.item_id)
                answer += item.weight * spent
    return answer


def _complete_block(catalog: Catalog, acquired: int, block: int) -> list[int]:
    unfinished = [p & ~acquired for item in catalog.workload
                  for p in catalog.supports[item.item_id] if p & ~acquired]
    indices = [i for i in range(len(catalog.ids)) if block & (1 << i)]
    return sorted(indices, key=lambda i: (
        -sum(bool(p & (1 << i)) for p in unfinished) / catalog.costs[i], catalog.ids[i]))


def committed_bundle_order(catalog: Catalog, *, max_local_union: int = 3,
                           max_bundle_candidates: int = 32) -> tuple[int, ...]:
    """Same candidate construction/density as LPBA, with block commitment.

    Candidate construction and caps deliberately match the frozen selector.
    This isolates replanning after one source from completing a selected block.
    """
    acquired = 0
    order: list[int] = []
    while acquired != catalog.relevant_mask:
        unfinished = [p & ~acquired for item in catalog.workload
                      for p in catalog.supports[item.item_id] if p & ~acquired]
        bundles = set(unfinished)
        for pos, left in enumerate(unfinished):
            for right in unfinished[pos + 1:]:
                union = left | right
                if union.bit_count() <= max_local_union:
                    bundles.add(union)
                if len(bundles) >= max_bundle_candidates * 4:
                    break
            if len(bundles) >= max_bundle_candidates * 4:
                break
        if not bundles:
            break
        before = catalog.value(acquired)
        block = min(bundles, key=lambda p: (
            -(catalog.value(acquired | p) - before) /
            sum(c for i, c in enumerate(catalog.costs) if p & (1 << i)),
            p.bit_count(), tuple(catalog.ids[i] for i in range(len(catalog.ids)) if p & (1 << i))))
        for index in _complete_block(catalog, acquired, block):
            order.append(index)
            acquired |= 1 << index
    return _finish_order(catalog, order)


def exact_density_order(catalog: Catalog) -> tuple[int, ...]:
    """Exponential exact-density greedy control on the relevant source set."""
    indices = [i for i in range(len(catalog.ids)) if catalog.relevant_mask & (1 << i)]
    count = len(indices)
    if count > 20:
        raise ValueError("exact-density resource cap exceeded")
    masks = np.arange(1 << count, dtype=np.uint32)
    costs = np.zeros(len(masks))
    value = np.zeros(len(masks))
    for k, index in enumerate(indices):
        costs += catalog.costs[index] * ((masks >> k) & 1)
    for item in catalog.workload:
        covered = np.zeros(len(masks), dtype=bool)
        for support in catalog.supports[item.item_id]:
            local = sum(1 << k for k, index in enumerate(indices) if support & (1 << index))
            covered |= (masks & local) == local
        value += item.weight * covered
    acquired = 0
    global_acquired = 0
    order: list[int] = []
    while acquired != len(masks) - 1:
        additions = masks[(masks != 0) & ((masks & acquired) == 0)]
        density = (value[additions | acquired] - value[acquired]) / costs[additions]
        maximum = float(np.max(density))
        if maximum <= 0:
            break
        ties = additions[np.isclose(density, maximum, rtol=0, atol=1e-12)]
        selected = min((int(p) for p in ties), key=lambda p: (
            p.bit_count(), tuple(catalog.ids[index] for k, index in enumerate(indices) if p & (1 << k))))
        block = sum(1 << index for k, index in enumerate(indices) if selected & (1 << k))
        for index in _complete_block(catalog, global_acquired, block):
            order.append(index)
            global_acquired |= 1 << index
        acquired |= selected
    return _finish_order(catalog, order)


def brute_force_optimal_latency(catalog: Catalog) -> float:
    """Independent backward DP, using Python sets and completion increments.

    Distinct from the frozen forward area DP. For audit-scale tests only.
    """
    indices = [i for i in range(len(catalog.ids)) if catalog.relevant_mask & (1 << i)]
    if len(indices) > 12:
        raise ValueError("independent Python audit DP cap exceeded")
    states = 1 << len(indices)
    values: list[float] = []
    costs: list[float] = []
    for local in range(states):
        global_mask = sum(1 << i for k, i in enumerate(indices) if local & (1 << k))
        values.append(catalog.value(global_mask))
        costs.append(sum(catalog.costs[i] for k, i in enumerate(indices) if local & (1 << k)))
    # Last acquisition charges its newly completed work at the prefix cost.
    dp = [0.0] + [float("inf")] * (states - 1)
    for mask in range(1, states):
        dp[mask] = min(dp[mask ^ (1 << k)] + costs[mask] *
                       (values[mask] - values[mask ^ (1 << k)])
                       for k in range(len(indices)) if mask & (1 << k))
    return dp[-1]
