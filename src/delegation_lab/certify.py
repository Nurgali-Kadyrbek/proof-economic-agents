"""Proof objects, a bounded Horn prover, and an independent replay checker."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from itertools import product
from typing import Any, Iterable, Mapping

from .evidence import EvidenceKind, EvidenceStore
from .ir import HornRule, Literal, Region, ToolProgram


@dataclass(frozen=True, slots=True)
class ProofNode:
    conclusion: Literal
    source_id: str | None = None
    rule_id: str | None = None
    premises: tuple["ProofNode", ...] = ()

    @property
    def support(self) -> frozenset[str]:
        own = {source for source in (self.source_id, self.rule_id) if source}
        return frozenset(own).union(*(premise.support for premise in self.premises))

    def as_dict(self) -> dict[str, Any]:
        return {
            "conclusion": self.conclusion.as_dict(),
            "source_id": self.source_id,
            "rule_id": self.rule_id,
            "premises": [premise.as_dict() for premise in self.premises],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProofNode":
        return cls(
            Literal.from_dict(value["conclusion"]),
            value.get("source_id"),
            value.get("rule_id"),
            tuple(cls.from_dict(item) for item in value.get("premises", [])),
        )


@dataclass(frozen=True, slots=True)
class Certificate:
    """A replayable delegation certificate, not an executable callback."""

    certificate_id: str
    region: Region
    program: ToolProgram
    claim: Literal
    proof: ProofNode
    support: frozenset[str]
    scope_dependencies: frozenset[str]
    closure_dependencies: frozenset[str]
    source_versions: tuple[tuple[str, str], ...]
    checker_version: str = "horn-replay-v1"

    @classmethod
    def build(
        cls,
        region: Region,
        program: ToolProgram,
        claim: Literal,
        proof: ProofNode,
        store: EvidenceStore,
        extra_scope_dependencies: Iterable[str] = (),
        closure_dependencies: Iterable[str] = (),
    ) -> "Certificate":
        support = proof.support
        records = [store.get(record_id) for record_id in sorted(support)]
        if any(record is None for record in records):
            raise ValueError("proof references evidence absent from store")
        resolved = [record for record in records if record is not None]
        source_versions = tuple((record.id, record.provenance.source_version) for record in resolved)
        scope = frozenset(extra_scope_dependencies).union(
            *(record.scope_dependencies for record in resolved)
        )
        closure = frozenset(closure_dependencies).union(
            *(record.closure_dependencies for record in resolved)
        )
        fingerprint = repr((region, program, claim, proof.as_dict(), source_versions, sorted(scope), sorted(closure)))
        return cls(
            certificate_id=sha256(fingerprint.encode()).hexdigest()[:20],
            region=region,
            program=program,
            claim=claim,
            proof=proof,
            support=support,
            scope_dependencies=scope,
            closure_dependencies=closure,
            source_versions=source_versions,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "certificate_id": self.certificate_id,
            "region": self.region.as_dict(),
            "program": self.program.as_dict(),
            "claim": self.claim.as_dict(),
            "proof": self.proof.as_dict(),
            "support": sorted(self.support),
            "scope_dependencies": sorted(self.scope_dependencies),
            "closure_dependencies": sorted(self.closure_dependencies),
            "source_versions": list(self.source_versions),
            "checker_version": self.checker_version,
        }


class ProofChecker:
    """Small trusted checker; it never consults model output or gold proofs."""

    version = "horn-replay-v1"

    def check(self, certificate: Certificate, store: EvidenceStore) -> tuple[bool, str]:
        if certificate.checker_version != self.version:
            return False, "checker version mismatch"
        if not certificate.support:
            return False, "empty support"
        if certificate.proof.conclusion != certificate.claim:
            return False, "proof conclusion does not equal claim"
        expected_versions: dict[str, str] = dict(certificate.source_versions)
        for record_id in certificate.support:
            record = store.get(record_id)
            if record is None:
                return False, f"missing support {record_id}"
            if expected_versions.get(record_id) != record.provenance.source_version:
                return False, f"version mismatch for {record_id}"
        valid, message, support = self._check_node(certificate.proof, store)
        if not valid:
            return False, message
        if support != certificate.support:
            return False, "declared support is not exact proof support"
        if not certificate.program.tool_name:
            return False, "program has no tool name"
        if any(atom.is_ground() is False for atom in certificate.program.local_contract):
            return False, "program local contract contains variables"
        return True, "accepted"

    def _check_node(self, node: ProofNode, store: EvidenceStore) -> tuple[bool, str, frozenset[str]]:
        if (node.source_id is None) == (node.rule_id is None):
            return False, "node must be either a fact or a rule application", frozenset()
        if node.source_id:
            record = store.get(node.source_id)
            if (
                record is None
                or record.kind is not EvidenceKind.FACT
                or not isinstance(record.payload, Literal)
                or record.payload != node.conclusion
                or node.premises
            ):
                return False, f"invalid fact node {node.source_id}", frozenset()
            return True, "accepted", frozenset({node.source_id})
        record = store.get(node.rule_id or "")
        if record is None or record.kind is not EvidenceKind.RULE or not isinstance(record.payload, HornRule):
            return False, f"missing rule {node.rule_id}", frozenset()
        if len(node.premises) != len(record.payload.body):
            return False, "wrong premise count", frozenset()
        bindings: dict[str, str] = {}
        support = {node.rule_id or ""}
        for template, premise in zip(record.payload.body, node.premises, strict=True):
            valid, message, premise_support = self._check_node(premise, store)
            if not valid:
                return False, message, frozenset()
            bindings = template.matches(premise.conclusion, bindings) or {}
            if not bindings and template != premise.conclusion:
                return False, "premise does not unify with rule body", frozenset()
            support.update(premise_support)
        if record.payload.head.substitute(bindings) != node.conclusion:
            return False, "rule head does not produce claimed conclusion", frozenset()
        return True, "accepted", frozenset(support)


def _bounded_append(existing: list[ProofNode], candidate: ProofNode, limit: int) -> bool:
    signature = tuple(sorted(candidate.support))
    if any(tuple(sorted(node.support)) == signature for node in existing):
        return False
    existing.append(candidate)
    existing.sort(key=lambda node: (len(node.support), tuple(sorted(node.support))))
    del existing[limit:]
    return candidate in existing


class HornProver:
    """Finite forward proof search retaining a small set of alternate supports."""

    def __init__(self, max_alternatives: int = 8, max_rounds: int = 32) -> None:
        self.max_alternatives = max_alternatives
        self.max_rounds = max_rounds

    def all_derivations(self, store: EvidenceStore) -> dict[Literal, tuple[ProofNode, ...]]:
        known: dict[Literal, list[ProofNode]] = {}
        for source_id, fact in store.facts():
            _bounded_append(known.setdefault(fact, []), ProofNode(fact, source_id=source_id), self.max_alternatives)
        for _ in range(self.max_rounds):
            changed = False
            snapshot = {literal: tuple(nodes) for literal, nodes in known.items()}
            for rule_source_id, rule in store.rules():
                for bindings, premises in self._instantiate_body(rule.body, snapshot):
                    conclusion = rule.head.substitute(bindings)
                    if not conclusion.is_ground():
                        continue
                    node = ProofNode(conclusion, rule_id=rule_source_id, premises=premises)
                    changed |= _bounded_append(known.setdefault(conclusion, []), node, self.max_alternatives)
            if not changed:
                break
        return {literal: tuple(nodes) for literal, nodes in known.items()}

    def prove(self, target: Literal, store: EvidenceStore) -> tuple[ProofNode, ...]:
        return self.all_derivations(store).get(target, ())

    def _instantiate_body(
        self, body: tuple[Literal, ...], known: Mapping[Literal, tuple[ProofNode, ...]]
    ) -> Iterable[tuple[dict[str, str], tuple[ProofNode, ...]]]:
        choices: list[list[tuple[dict[str, str], ProofNode]]] = []
        # Progressive matching preserves variable correlation across conjunctions.
        partials: list[tuple[dict[str, str], tuple[ProofNode, ...]]] = [({}, ())]
        for template in body:
            next_partials: list[tuple[dict[str, str], tuple[ProofNode, ...]]] = []
            for bindings, premises in partials:
                for literal, derivations in known.items():
                    matched = template.matches(literal, bindings)
                    if matched is None:
                        continue
                    for derivation in derivations:
                        next_partials.append((matched, premises + (derivation,)))
            partials = next_partials[: self.max_alternatives * 64]
            if not partials:
                return ()
        return partials
