"""Versioned evidence, provenance, and the trusted/untrusted boundary."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable

from .ir import HornRule, Literal


class TrustLevel(str, Enum):
    PROPOSED = "proposed"
    ACCEPTED = "accepted"
    QUARANTINED = "quarantined"


class EvidenceKind(str, Enum):
    FACT = "fact"
    RULE = "rule"
    POLICY = "policy"
    SCOPE = "scope"
    STATE = "state"


@dataclass(frozen=True, slots=True)
class Provenance:
    source_uri: str
    source_version: str
    source_sha256: str
    authority: str
    retrieved_at: str = ""
    proposed_by: str | None = None

    @property
    def is_model_proposal(self) -> bool:
        return self.proposed_by is not None


@dataclass(frozen=True, slots=True)
class EvidenceKey:
    source_id: str
    item_id: str

    def __str__(self) -> str:
        return f"{self.source_id}:{self.item_id}"


EvidencePayload = Literal | HornRule | dict[str, Any]


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    key: EvidenceKey
    kind: EvidenceKind
    payload: EvidencePayload
    provenance: Provenance
    cost: float = 1.0
    trust: TrustLevel = TrustLevel.PROPOSED
    scope_dependencies: frozenset[str] = frozenset()
    closure_dependencies: frozenset[str] = frozenset()

    @property
    def id(self) -> str:
        return str(self.key)

    def accepted(self) -> "EvidenceRecord":
        return EvidenceRecord(
            self.key,
            self.kind,
            self.payload,
            self.provenance,
            self.cost,
            TrustLevel.ACCEPTED,
            self.scope_dependencies,
            self.closure_dependencies,
        )


class EvidenceStore:
    """Only an authoritative validator can add usable evidence.

    A model-generated object may be kept as a proposal for audit, but calling
    :meth:`admit` on it raises.  This makes accidental trust escalation visible
    at the small trusted-core boundary.
    """

    def __init__(self, records: Iterable[EvidenceRecord] = ()) -> None:
        self._accepted: dict[str, EvidenceRecord] = {}
        self.proposals: dict[str, EvidenceRecord] = {}
        self.quarantined: dict[str, EvidenceRecord] = {}
        for record in records:
            self.admit(record)

    def copy(self) -> "EvidenceStore":
        return EvidenceStore(self._accepted.values())

    @property
    def records(self) -> tuple[EvidenceRecord, ...]:
        return tuple(self._accepted[key] for key in sorted(self._accepted))

    @property
    def ids(self) -> frozenset[str]:
        return frozenset(self._accepted)

    def get(self, record_id: str) -> EvidenceRecord | None:
        return self._accepted.get(record_id)

    def propose(self, record: EvidenceRecord) -> None:
        self.proposals[record.id] = record

    def admit(self, record: EvidenceRecord) -> None:
        if record.trust is not TrustLevel.ACCEPTED:
            raise ValueError(f"evidence {record.id} is not marked accepted")
        if record.provenance.is_model_proposal:
            raise ValueError(f"model proposal {record.id} needs independent authoritative validation")
        if record.kind is EvidenceKind.RULE:
            if not isinstance(record.payload, HornRule) or not record.payload.is_safe():
                raise ValueError(f"unsafe or malformed rule {record.id}")
        if record.kind is EvidenceKind.FACT and not isinstance(record.payload, Literal):
            raise ValueError(f"fact {record.id} has non-literal payload")
        conflict = self._find_conflict(record)
        if conflict:
            self.quarantined[record.id] = record
            raise ValueError(f"inconsistent evidence {record.id} conflicts with {conflict.id}")
        self._accepted[record.id] = record

    def _find_conflict(self, candidate: EvidenceRecord) -> EvidenceRecord | None:
        if candidate.kind is not EvidenceKind.FACT or not isinstance(candidate.payload, Literal):
            return None
        for existing in self._accepted.values():
            if (
                existing.kind is EvidenceKind.FACT
                and isinstance(existing.payload, Literal)
                and existing.payload == candidate.payload.negate()
            ):
                return existing
        return None

    def facts(self) -> tuple[tuple[str, Literal], ...]:
        return tuple(
            (record.id, record.payload)
            for record in self.records
            if record.kind is EvidenceKind.FACT and isinstance(record.payload, Literal)
        )

    def rules(self) -> tuple[tuple[str, HornRule], ...]:
        return tuple(
            (record.id, record.payload)
            for record in self.records
            if record.kind is EvidenceKind.RULE and isinstance(record.payload, HornRule)
        )
