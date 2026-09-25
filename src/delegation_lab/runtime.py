"""Guarded deterministic execution with a small atomic-commit abstraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, Mapping, Protocol

from .certify import Certificate, ProofChecker
from .evidence import EvidenceStore
from .ir import Literal
from .lifecycle import CertificateRegistry


@dataclass(frozen=True, slots=True)
class RequestContext:
    request_id: str
    principal: str
    consent_hash: str
    fields: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class Snapshot:
    state_version: str
    policy_epochs: Mapping[str, str]
    facts: frozenset[Literal]


@dataclass(frozen=True, slots=True)
class ExecutionReceipt:
    certificate_id: str
    request_hash: str
    principal: str
    consent_hash: str
    read_version: str
    policy_epochs: tuple[tuple[str, str], ...]
    idempotency_key: str


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    delegated: bool
    committed: bool
    reason: str
    receipt: ExecutionReceipt | None = None
    result: Any = None


class TransactionalToolBackend(Protocol):
    def snapshot(self) -> Snapshot: ...

    def is_write(self, tool_name: str) -> bool: ...

    def commit(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
        *,
        expected_state_version: str,
        expected_policy_epochs: Mapping[str, str],
        idempotency_key: str,
    ) -> tuple[bool, Any, str]: ...


class CertificateRuntime:
    """Routes only active, checker-accepted certificates through atomic guards."""

    def __init__(self, registry: CertificateRegistry, evidence: EvidenceStore, backend: TransactionalToolBackend) -> None:
        self.registry = registry
        self.evidence = evidence
        self.backend = backend
        self.checker = ProofChecker()

    def execute(self, context: RequestContext) -> ExecutionResult:
        snapshot = self.backend.snapshot()
        for certificate in self.registry.active():
            if not certificate.region.matches(context.fields):
                continue
            valid, message = self.checker.check(certificate, self.evidence)
            if not valid:
                self.registry.suspend(certificate.certificate_id, message)
                continue
            if not all(contract in snapshot.facts for contract in certificate.program.instantiate_contract(context.fields)):
                continue
            expected_epochs = {dependency: snapshot.policy_epochs.get(dependency, "") for dependency in certificate.scope_dependencies}
            if any(not epoch for epoch in expected_epochs.values()):
                continue
            idempotency_key = sha256(f"{certificate.certificate_id}:{context.request_id}".encode()).hexdigest()
            receipt = ExecutionReceipt(
                certificate.certificate_id,
                sha256(context.request_id.encode()).hexdigest(),
                context.principal,
                context.consent_hash,
                snapshot.state_version,
                tuple(sorted(expected_epochs.items())),
                idempotency_key,
            )
            committed, result, reason = self.backend.commit(
                certificate.program.tool_name,
                certificate.program.instantiate(context.fields),
                expected_state_version=snapshot.state_version,
                expected_policy_epochs=expected_epochs,
                idempotency_key=idempotency_key,
            )
            return ExecutionResult(True, committed, reason, receipt, result)
        return ExecutionResult(False, False, "no matching active certificate")


@dataclass
class InMemoryBackend:
    """Test/smoke backend proving update-before-commit and retry semantics."""

    facts: set[Literal] = field(default_factory=set)
    state_version: int = 0
    policy_epochs: dict[str, str] = field(default_factory=dict)
    effects: dict[str, Any] = field(default_factory=dict)
    tool_mutability: dict[str, bool] = field(default_factory=dict)

    def is_write(self, tool_name: str) -> bool:
        return self.tool_mutability.get(tool_name, True)

    def snapshot(self) -> Snapshot:
        return Snapshot(str(self.state_version), dict(self.policy_epochs), frozenset(self.facts))

    def change(self, *, facts: set[Literal] | None = None, dependency: str | None = None, epoch: str | None = None) -> None:
        if facts is not None:
            self.facts = set(facts)
        if dependency is not None:
            self.policy_epochs[dependency] = epoch or str(int(self.policy_epochs.get(dependency, "0")) + 1)
        self.state_version += 1

    def commit(self, tool_name: str, arguments: Mapping[str, Any], *, expected_state_version: str, expected_policy_epochs: Mapping[str, str], idempotency_key: str) -> tuple[bool, Any, str]:
        if idempotency_key in self.effects:
            return True, self.effects[idempotency_key], "idempotent replay"
        if str(self.state_version) != expected_state_version:
            return False, None, "state changed before commit"
        if any(self.policy_epochs.get(key) != epoch for key, epoch in expected_policy_epochs.items()):
            return False, None, "policy changed before commit"
        effect = {"tool": tool_name, "arguments": dict(arguments)}
        self.effects[idempotency_key] = effect
        self.state_version += 1
        return True, effect, "committed"
