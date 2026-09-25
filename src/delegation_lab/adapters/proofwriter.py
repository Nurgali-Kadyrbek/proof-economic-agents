"""Adapter for the original ProofWriter V2020.12.3 OWA JSONL release."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import urllib.request
import zipfile
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

from ..acquire import WorkItem
from ..evidence import EvidenceKey, EvidenceKind, EvidenceRecord, Provenance, TrustLevel
from ..ir import HornRule, Literal
from ..llm import ChatMessage, LLMAdapter, ToolSchema


PROOFWRITER_URL = "https://aristo-data-public.s3.amazonaws.com/proofwriter/proofwriter-dataset-V2020.12.3.zip"
PROOFWRITER_ARCHIVE_SHA256 = "bbc5694901e8306d0bd659aa1ad53ccfd02c201864f4b320ffa3777827d1fc26"
PROOFWRITER_RELEASE = "V2020.12.3"
_TUPLE_PATTERN = re.compile(r'\(\s*"([^"]+)"\s+"([^"]+)"\s+"([^"]+)"\s+"([+~-])"\s*\)')


@dataclass(frozen=True, slots=True)
class ProofWriterTheory:
    theory_id: str
    split: str
    depth: str
    candidates: tuple[EvidenceRecord, ...]
    workload: tuple[WorkItem, ...]
    source_text: tuple[tuple[str, str], ...]


def download_proofwriter(destination: Path) -> Path:
    """Fetch the original release and verify the archive before extracting it."""
    destination = destination.resolve()
    source_root = destination / f"proofwriter-dataset-{PROOFWRITER_RELEASE}"
    if source_root.exists():
        return source_root
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / f"proofwriter-dataset-{PROOFWRITER_RELEASE}.zip"
    if not archive.exists():
        with urllib.request.urlopen(PROOFWRITER_URL) as response, archive.open("wb") as output:
            shutil.copyfileobj(response, output)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != PROOFWRITER_ARCHIVE_SHA256:
        raise RuntimeError(f"ProofWriter checksum mismatch: {digest}")
    with zipfile.ZipFile(archive) as zipped:
        zipped.extractall(destination)
    if not source_root.exists():
        raise RuntimeError("ProofWriter archive did not contain the expected release root")
    return source_root


def load_proofwriter_theories(root: Path, *, split: str = "test", depth: str = "depth-5", limit: int | None = None) -> tuple[ProofWriterTheory, ...]:
    """Load original grouped theories without reading gold proof DAGs for policy."""
    root = root.resolve()
    if (root / f"proofwriter-dataset-{PROOFWRITER_RELEASE}").exists():
        root = root / f"proofwriter-dataset-{PROOFWRITER_RELEASE}"
    data_path = root / "OWA" / depth / f"meta-{split}.jsonl"
    if not data_path.exists():
        raise FileNotFoundError(f"Original ProofWriter OWA split not found: {data_path}")
    theories: list[ProofWriterTheory] = []
    with data_path.open(encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            theories.append(_parse_theory(raw, split=split, depth=depth))
            if limit is not None and len(theories) >= limit:
                break
    return tuple(theories)


def _parse_literal(representation: str, namespace: str = "") -> Literal:
    tuples = _TUPLE_PATTERN.findall(representation)
    if len(tuples) != 1:
        raise ValueError(f"expected one literal in representation: {representation}")
    subject, relation, object_, polarity = tuples[0]
    terms = tuple(_namespace(_variable(term), namespace) for term in (subject, object_))
    return Literal(relation, terms, polarity == "+")


def _parse_rule(rule_id: str, representation: str, namespace: str = "") -> HornRule:
    tuples = _TUPLE_PATTERN.findall(representation)
    if len(tuples) < 2:
        raise ValueError(f"expected a nonempty Horn rule: {representation}")
    literals = tuple(
        # In the original OWA formal representation, `~` denotes a negative
        # body condition while `-` denotes an explicit negative literal.
        Literal(relation, tuple(_namespace(_variable(term), namespace) for term in (subject, object_)), polarity == "+")
        for subject, relation, object_, polarity in tuples
    )
    return HornRule(rule_id, literals[:-1], literals[-1])


def _variable(term: str) -> str:
    # ProofWriter uses "something" as the only quantified variable marker.
    return "?something" if term == "something" else term


def _namespace(term: str, namespace: str) -> str:
    return term if term.startswith("?") or not namespace else f"{namespace}/{term}"


def _parse_theory(raw: dict[str, Any], *, split: str, depth: str) -> ProofWriterTheory:
    theory_id = str(raw["id"])
    provenance = Provenance(
        source_uri=PROOFWRITER_URL,
        source_version=PROOFWRITER_RELEASE,
        source_sha256=PROOFWRITER_ARCHIVE_SHA256,
        authority="AI2 ProofWriter release",
    )
    scope_id = f"proofwriter:{theory_id}:closure"
    candidates: list[EvidenceRecord] = []
    source_text: list[tuple[str, str]] = []
    for item_id, payload in raw["triples"].items():
        source_text.append((item_id, str(payload["text"])))
        candidates.append(
            EvidenceRecord(
                EvidenceKey(theory_id, item_id),
                EvidenceKind.FACT,
                _parse_literal(payload["representation"], theory_id),
                provenance,
                cost=1.0,
                trust=TrustLevel.ACCEPTED,
                scope_dependencies=frozenset({scope_id}),
            )
        )
    for item_id, payload in raw["rules"].items():
        source_text.append((item_id, str(payload["text"])))
        rule = _parse_rule(item_id, payload["representation"], theory_id)
        candidates.append(
            EvidenceRecord(
                EvidenceKey(theory_id, item_id),
                EvidenceKind.RULE,
                rule,
                provenance,
                cost=1.0 + 0.2 * len(rule.body),
                trust=TrustLevel.ACCEPTED,
                scope_dependencies=frozenset({scope_id}),
            )
        )
    workload = tuple(
        WorkItem(
            item_id=f"{theory_id}:{question_id}",
            target=_parse_literal(question["representation"], theory_id),
            cluster_id=theory_id,
            expected_outcome=_outcome(question["answer"]),
        )
        for question_id, question in sorted(raw["questions"].items())
    )
    return ProofWriterTheory(theory_id, split, depth, tuple(candidates), workload, tuple(source_text))


def _outcome(value: Any) -> str:
    if value is True:
        return "entailed"
    if value is False:
        return "contradicted"
    return "unknown"


def combine_theories(theories: Iterable[ProofWriterTheory]) -> tuple[tuple[EvidenceRecord, ...], tuple[WorkItem, ...]]:
    candidates: list[EvidenceRecord] = []
    workload: list[WorkItem] = []
    for theory in theories:
        candidates.extend(theory.candidates)
        workload.extend(theory.workload)
    return tuple(candidates), tuple(workload)


def predicate_frequency_prior(theories: Iterable[ProofWriterTheory]) -> dict[str, float]:
    """Estimate a workload prior from query inputs alone, never answers."""
    counts = Counter(item.target.predicate for theory in theories for item in theory.workload)
    total = sum(counts.values())
    if not total:
        raise ValueError("cannot build a workload prior from an empty query set")
    return {predicate: count / total for predicate, count in sorted(counts.items())}


def reweight_theories_by_predicate_prior(
    theories: Iterable[ProofWriterTheory], prior: dict[str, float]
) -> tuple[ProofWriterTheory, ...]:
    """Give recurrent query classes their train-estimated demand weights.

    Per-theory normalization preserves each original theory's aggregate weight,
    so cluster bootstrap remains a matched comparison rather than being driven
    by the number of questions in a theory.  A missing predicate is an error:
    silently assigning it a weight would make the workload definition drift.
    """
    reweighted: list[ProofWriterTheory] = []
    for theory in theories:
        raw = [prior.get(item.target.predicate) for item in theory.workload]
        if any(value is None for value in raw):
            missing = sorted({item.target.predicate for item, value in zip(theory.workload, raw) if value is None})
            raise ValueError(f"workload prior has no mass for predicates: {missing}")
        mean = sum(float(value) for value in raw) / len(raw) if raw else 0.0
        if mean <= 0:
            raise ValueError("workload prior has zero mean for a theory")
        workload = tuple(replace(item, weight=float(value) / mean) for item, value in zip(theory.workload, raw))
        reweighted.append(replace(theory, workload=workload))
    return tuple(reweighted)


def qwen_propose_formal_items(adapter: LLMAdapter, theory: ProofWriterTheory) -> tuple[EvidenceRecord, ...]:
    """Ask a model to translate visible source text into *untrusted* IR proposals.

    The result is intentionally unusable by :class:`EvidenceStore`. Call
    :func:`validate_proposed_items` to match it against an authoritative source
    representation before acquisition; a lexical model proposal is never a fact.
    """
    sources = "\n".join(f"{item_id}: {text}" for item_id, text in theory.source_text)
    schema = ToolSchema(
        "submit_formalization",
        "Submit only candidate formalizations; they will be independently checked.",
        {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "item_id": {"type": "string"},
                            "kind": {"enum": ["fact", "rule"]},
                            "literal": {"type": "object"},
                            "rule": {"type": "object"},
                        },
                        "required": ["item_id", "kind"],
                    },
                }
            },
            "required": ["items"],
        },
    )
    output = adapter.complete(
        (
            ChatMessage("system", "Translate the given synthetic source statements to typed Horn IR. Do not answer questions."),
            ChatMessage("user", sources),
        ),
        (schema,),
    )
    submission = next((call for call in output.tool_calls if call.name == "submit_formalization"), None)
    if submission is None:
        return ()
    provenance = Provenance(
        source_uri=f"model://{adapter.model_id}",
        source_version=adapter.model_revision,
        source_sha256="untrusted-model-output",
        authority="model proposal",
        proposed_by=adapter.model_id,
    )
    proposals: list[EvidenceRecord] = []
    for item in submission.arguments.get("items", []):
        try:
            item_id, kind = str(item["item_id"]), str(item["kind"])
            if kind == "fact":
                payload = Literal.from_dict(item["literal"])
                evidence_kind = EvidenceKind.FACT
            elif kind == "rule":
                payload = HornRule.from_dict(item["rule"])
                evidence_kind = EvidenceKind.RULE
            else:
                continue
            proposals.append(EvidenceRecord(EvidenceKey(theory.theory_id, item_id), evidence_kind, payload, provenance, trust=TrustLevel.PROPOSED))
        except (KeyError, TypeError, ValueError):
            continue
    return tuple(proposals)


def validate_proposed_items(proposals: Iterable[EvidenceRecord], authoritative_candidates: Iterable[EvidenceRecord]) -> tuple[EvidenceRecord, ...]:
    """Return only exact source-backed matches; proposals themselves are discarded."""
    authoritative = {record.id: record for record in authoritative_candidates}
    validated: list[EvidenceRecord] = []
    for proposal in proposals:
        source = authoritative.get(proposal.id)
        if source and source.kind is proposal.kind and source.payload == proposal.payload:
            validated.append(source)
    return tuple(validated)
