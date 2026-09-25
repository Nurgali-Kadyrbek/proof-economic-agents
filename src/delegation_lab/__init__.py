"""Small, model-independent research toolkit for certified delegation."""

from .acquire import AcquisitionMethod, AcquisitionRun
from .certify import Certificate, HornProver, ProofChecker
from .evidence import EvidenceRecord, EvidenceStore, Provenance, TrustLevel
from .ir import HornRule, Literal, ToolProgram
from .proof_economics import (
    ControllerObjective,
    EvidenceSelectionPolicy,
    LifecycleBenefitLowerBound,
    ProofEconomicDecision,
    ProofEconomicDelegationController,
    ProofEconomicStrategy,
    StructuralWorkloadFeatures,
    TimingPolicy,
    structural_features_from_catalog,
)

__all__ = [
    "AcquisitionMethod",
    "AcquisitionRun",
    "Certificate",
    "ControllerObjective",
    "EvidenceRecord",
    "EvidenceSelectionPolicy",
    "EvidenceStore",
    "HornProver",
    "HornRule",
    "Literal",
    "LifecycleBenefitLowerBound",
    "ProofChecker",
    "ProofEconomicDecision",
    "ProofEconomicDelegationController",
    "ProofEconomicStrategy",
    "Provenance",
    "StructuralWorkloadFeatures",
    "ToolProgram",
    "TimingPolicy",
    "TrustLevel",
    "structural_features_from_catalog",
]
