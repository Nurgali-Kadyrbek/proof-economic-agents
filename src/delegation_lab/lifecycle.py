"""Dependency-footprint invalidation for reusable certificates."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from .certify import Certificate, ProofChecker
from .evidence import EvidenceStore


class CertificateStatus(str, Enum):
    ACTIVE = "active"
    SUSPENDED = "suspended"
    RETIRED = "retired"


@dataclass(slots=True)
class RegisteredCertificate:
    certificate: Certificate
    status: CertificateStatus = CertificateStatus.ACTIVE
    suspension_reason: str | None = None


class CertificateRegistry:
    """A reverse index over positive, scope, and closure dependencies.

    This deliberately indexes more than positive proof leaves. A new exception or
    changed completeness manifest invalidates relevant authority even when no old
    leaf fact was edited.
    """

    def __init__(self, checker: ProofChecker | None = None) -> None:
        self.checker = checker or ProofChecker()
        self._certificates: dict[str, RegisteredCertificate] = {}
        self._reverse: dict[str, set[str]] = {}

    def register(self, certificate: Certificate, store: EvidenceStore) -> None:
        valid, message = self.checker.check(certificate, store)
        if not valid:
            raise ValueError(f"certificate rejected: {message}")
        if certificate.certificate_id in self._certificates:
            raise ValueError(f"duplicate certificate {certificate.certificate_id}")
        self._certificates[certificate.certificate_id] = RegisteredCertificate(certificate)
        for dependency in self.dependencies(certificate):
            self._reverse.setdefault(dependency, set()).add(certificate.certificate_id)

    @staticmethod
    def dependencies(certificate: Certificate) -> frozenset[str]:
        return certificate.support.union(certificate.scope_dependencies, certificate.closure_dependencies)

    def active(self) -> tuple[Certificate, ...]:
        return tuple(
            registered.certificate
            for registered in self._certificates.values()
            if registered.status is CertificateStatus.ACTIVE
        )

    def status(self, certificate_id: str) -> CertificateStatus:
        return self._certificates[certificate_id].status

    def suspend_changed(self, dependency_ids: Iterable[str], reason: str = "dependency changed") -> tuple[str, ...]:
        affected: set[str] = set()
        for dependency in dependency_ids:
            affected.update(self._reverse.get(dependency, set()))
        for certificate_id in affected:
            registered = self._certificates[certificate_id]
            if registered.status is CertificateStatus.ACTIVE:
                registered.status = CertificateStatus.SUSPENDED
                registered.suspension_reason = reason
        return tuple(sorted(affected))

    def suspend(self, certificate_id: str, reason: str) -> None:
        registered = self._certificates[certificate_id]
        registered.status = CertificateStatus.SUSPENDED
        registered.suspension_reason = reason

    def revalidate(self, certificate_id: str, store: EvidenceStore) -> bool:
        registered = self._certificates[certificate_id]
        valid, message = self.checker.check(registered.certificate, store)
        if valid:
            registered.status = CertificateStatus.ACTIVE
            registered.suspension_reason = None
            return True
        registered.status = CertificateStatus.SUSPENDED
        registered.suspension_reason = message
        return False
