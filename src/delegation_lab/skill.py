"""Bounded, proof replayed sequences of guarded tool calls.

The program is data, not generated Python. Domain adapters supply facts from
tool observations; the trusted runtime only matches literals and commits one
guarded step at a time. Every write must be the final step.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Iterable, Mapping

from .certify import Certificate, ProofChecker
from .evidence import EvidenceStore
from .ir import Literal, Region
from .lifecycle import CertificateRegistry
from .runtime import ExecutionReceipt, ExecutionResult, RequestContext, TransactionalToolBackend


def step_claim(tool_name: str, program: Any, postconditions: tuple[Literal, ...] = (), writes: bool = False) -> Literal:
    """Bind proof to the exact tool, arguments, guards, postcheck, and effect type."""
    digest = sha256(repr((program, postconditions, writes)).encode()).hexdigest()[:20]
    return Literal("skill_step_contract", (tool_name, digest))


@dataclass(frozen=True, slots=True)
class SkillStep:
    certificate: Certificate
    postconditions: tuple[Literal, ...]
    writes: bool = False

    def postconditions_for(self, fields: Mapping[str, str]) -> tuple[Literal, ...]:
        return tuple(
            Literal(atom.predicate, tuple(fields[term[1:]] if term.startswith("$") else term for term in atom.terms), atom.positive)
            for atom in self.postconditions
        )


@dataclass(frozen=True, slots=True)
class CertifiedSkill:
    skill_id: str
    region: Region
    required_inputs: tuple[str, ...]
    preconditions: tuple[Literal, ...]
    steps: tuple[SkillStep, ...]
    support: frozenset[str]
    scope_dependencies: frozenset[str]
    closure_dependencies: frozenset[str]
    source_versions: tuple[tuple[str, str], ...]
    checker_version: str = "bounded-skill-v1"

    @classmethod
    def build(
        cls,
        region: Region,
        required_inputs: Iterable[str],
        preconditions: Iterable[Literal],
        steps: Iterable[SkillStep],
    ) -> "CertifiedSkill":
        bounded = tuple(steps)
        if not 2 <= len(bounded) <= 4:
            raise ValueError("a skill needs two to four steps")
        support = frozenset().union(*(step.certificate.support for step in bounded))
        scope = frozenset().union(*(step.certificate.scope_dependencies for step in bounded))
        closure = frozenset().union(*(step.certificate.closure_dependencies for step in bounded))
        versions = tuple(sorted(set().union(*(set(step.certificate.source_versions) for step in bounded))))
        inputs = tuple(sorted(set(required_inputs)))
        guards = tuple(preconditions)
        fingerprint = repr((region, inputs, guards, bounded, versions))
        return cls(sha256(fingerprint.encode()).hexdigest()[:20], region, inputs, guards, bounded, support, scope, closure, versions)

    @property
    def certificate_id(self) -> str:
        return self.skill_id

    def as_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "region": self.region.as_dict(),
            "required_inputs": list(self.required_inputs),
            "preconditions": [atom.as_dict() for atom in self.preconditions],
            "steps": [
                {
                    "certificate": step.certificate.as_dict(),
                    "postconditions": [atom.as_dict() for atom in step.postconditions],
                    "writes": step.writes,
                }
                for step in self.steps
            ],
            "support": sorted(self.support),
            "scope_dependencies": sorted(self.scope_dependencies),
            "closure_dependencies": sorted(self.closure_dependencies),
            "source_versions": list(self.source_versions),
            "checker_version": self.checker_version,
        }

    def preconditions_for(self, fields: Mapping[str, str]) -> tuple[Literal, ...]:
        return tuple(
            Literal(atom.predicate, tuple(fields[term[1:]] if term.startswith("$") else term for term in atom.terms), atom.positive)
            for atom in self.preconditions
        )


class SkillChecker:
    version = "bounded-skill-v1"

    def check(self, skill: CertifiedSkill, store: EvidenceStore) -> tuple[bool, str]:
        if skill.checker_version != self.version or not 2 <= len(skill.steps) <= 4:
            return False, "unsupported skill version or bound"
        if any(step.writes for step in skill.steps[:-1]):
            return False, "a write is allowed only as the final step"
        checker = ProofChecker()
        for step in skill.steps:
            valid, reason = checker.check(step.certificate, store)
            if not valid:
                return False, reason
            if step.certificate.region != skill.region:
                return False, "step applicability differs from skill applicability"
            if not step.certificate.program.tool_name:
                return False, "empty tool name"
            if step.certificate.claim != step_claim(step.certificate.program.tool_name, step.certificate.program, step.postconditions, step.writes):
                return False, "proof is not bound to the exact step program"
            if not step.postconditions:
                return False, "step has no checked postcondition"
        if skill.support != frozenset().union(*(step.certificate.support for step in skill.steps)):
            return False, "support does not match step proofs"
        if skill.scope_dependencies != frozenset().union(*(step.certificate.scope_dependencies for step in skill.steps)):
            return False, "scope does not match step proofs"
        if skill.closure_dependencies != frozenset().union(*(step.certificate.closure_dependencies for step in skill.steps)):
            return False, "closure does not match step proofs"
        versions = tuple(sorted(set().union(*(set(step.certificate.source_versions) for step in skill.steps))))
        if skill.source_versions != versions:
            return False, "source versions do not match step proofs"
        return True, "accepted"


class SkillRuntime:
    """Checked sequential execution with a fresh snapshot and CAS per step."""

    def __init__(self, registry: CertificateRegistry, evidence: EvidenceStore, backend: TransactionalToolBackend) -> None:
        self.registry = registry
        self.evidence = evidence
        self.backend = backend
        self.checker = SkillChecker()

    def execute(self, context: RequestContext) -> ExecutionResult:
        for skill in self.registry.active():
            if not isinstance(skill, CertifiedSkill) or not skill.region.matches(context.fields):
                continue
            if any(key not in context.fields for key in skill.required_inputs):
                continue
            if context.fields.get("principal") != context.principal:
                continue
            valid, reason = self.checker.check(skill, self.evidence)
            if not valid:
                self.registry.suspend(skill.skill_id, reason)
                continue
            first = self.backend.snapshot()
            if not all(atom in first.facts for atom in skill.preconditions_for(context.fields)):
                continue
            epochs = {dependency: first.policy_epochs.get(dependency, "") for dependency in skill.scope_dependencies}
            if any(not epoch for epoch in epochs.values()):
                continue
            effects: list[dict[str, Any]] = []
            receipt: ExecutionReceipt | None = None
            bindings = dict(context.fields)
            for index, step in enumerate(skill.steps):
                snapshot = self.backend.snapshot()
                if any(snapshot.policy_epochs.get(key) != value for key, value in epochs.items()):
                    return ExecutionResult(bool(effects), bool(effects), "dependency changed", receipt, effects)
                try:
                    arguments = step.certificate.program.instantiate(bindings)
                    contracts = step.certificate.program.instantiate_contract(bindings)
                    posts = step.postconditions_for(bindings)
                except KeyError:
                    return ExecutionResult(bool(effects), bool(effects), "missing structured input", receipt, effects)
                if bool(self.backend.is_write(step.certificate.program.tool_name)) != step.writes:
                    return ExecutionResult(bool(effects), bool(effects), "tool effect type disagrees with reviewed step", receipt, effects)
                if not all(atom in snapshot.facts for atom in contracts):
                    return ExecutionResult(bool(effects), bool(effects), "step precondition failed", receipt, effects)
                key = sha256(f"{skill.skill_id}:{context.request_id}:{index}".encode()).hexdigest()
                receipt = ExecutionReceipt(skill.skill_id, sha256(context.request_id.encode()).hexdigest(), context.principal, context.consent_hash, snapshot.state_version, tuple(sorted(epochs.items())), key)
                committed, effect, reason = self.backend.commit(
                    step.certificate.program.tool_name,
                    arguments,
                    expected_state_version=snapshot.state_version,
                    expected_policy_epochs=epochs,
                    idempotency_key=key,
                )
                if not committed:
                    return ExecutionResult(bool(effects), bool(effects), reason, receipt, effects)
                effects.append(effect)
                # A domain adapter may expose new structured fields obtained
                # from this trusted tool result. Existing input bindings can
                # never be overwritten during a skill execution.
                if hasattr(self.backend, "skill_bindings"):
                    for key, value in self.backend.skill_bindings().items():
                        bindings.setdefault(key, value)
                if not all(atom in self.backend.snapshot().facts for atom in posts):
                    return ExecutionResult(True, True, "step postcondition failed; stopped", receipt, effects)
            return ExecutionResult(True, True, "skill completed", receipt, effects)
        return ExecutionResult(False, False, "no matching active skill")
