"""The deliberately small, serializable deterministic program fragment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


def _substitute(value: Any, bindings: Mapping[str, str]) -> Any:
    if isinstance(value, str) and value.startswith("$"):
        return bindings[value[1:]]
    if isinstance(value, list):
        return [_substitute(item, bindings) for item in value]
    if isinstance(value, tuple):
        return tuple(_substitute(item, bindings) for item in value)
    if isinstance(value, dict):
        return {key: _substitute(item, bindings) for key, item in value.items()}
    return value


@dataclass(frozen=True, slots=True, order=True)
class Literal:
    """A signed Datalog atom. Variables are terms beginning with ``?``."""

    predicate: str
    terms: tuple[str, ...]
    positive: bool = True

    def negate(self) -> "Literal":
        return Literal(self.predicate, self.terms, not self.positive)

    def is_ground(self) -> bool:
        return all(not term.startswith("?") for term in self.terms)

    def substitute(self, bindings: Mapping[str, str]) -> "Literal":
        return Literal(
            self.predicate,
            tuple(bindings.get(term, term) for term in self.terms),
            self.positive,
        )

    def matches(self, ground: "Literal", bindings: Mapping[str, str]) -> dict[str, str] | None:
        """Unify this (normally rule-side) atom with a ground atom."""
        if self.predicate != ground.predicate or self.positive != ground.positive:
            return None
        if len(self.terms) != len(ground.terms):
            return None
        result = dict(bindings)
        for left, right in zip(self.terms, ground.terms, strict=True):
            if left.startswith("?"):
                old = result.get(left)
                if old is not None and old != right:
                    return None
                result[left] = right
            elif left != right:
                return None
        return result

    def as_dict(self) -> dict[str, Any]:
        return {"predicate": self.predicate, "terms": list(self.terms), "positive": self.positive}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Literal":
        return cls(str(value["predicate"]), tuple(value["terms"]), bool(value.get("positive", True)))


@dataclass(frozen=True, slots=True)
class HornRule:
    """A safe Horn rule. The checker rejects unbound head variables."""

    rule_id: str
    body: tuple[Literal, ...]
    head: Literal

    def is_safe(self) -> bool:
        body_variables = {term for atom in self.body for term in atom.terms if term.startswith("?")}
        return all(not term.startswith("?") or term in body_variables for term in self.head.terms)

    def as_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "body": [atom.as_dict() for atom in self.body],
            "head": self.head.as_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "HornRule":
        return cls(
            str(value["rule_id"]),
            tuple(Literal.from_dict(atom) for atom in value["body"]),
            Literal.from_dict(value["head"]),
        )


@dataclass(frozen=True, slots=True)
class Region:
    """A conjunction of equalities over a normalized request/state envelope."""

    equals: tuple[tuple[str, str], ...] = ()

    def matches(self, fields: Mapping[str, Any]) -> bool:
        return all(str(fields.get(key)) == expected for key, expected in self.equals)

    def as_dict(self) -> dict[str, Any]:
        return {"equals": list(self.equals)}


@dataclass(frozen=True, slots=True)
class ToolProgram:
    """An acyclic deterministic program: exactly one permitted tool invocation.

    Arguments can reference normalized request fields with ``$field_name``.  There
    is intentionally no Python callback, shell command, model call, or eval path.
    """

    tool_name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    local_contract: tuple[Literal, ...] = ()

    def instantiate(self, fields: Mapping[str, str]) -> dict[str, Any]:
        return {key: _substitute(value, fields) for key, value in self.arguments.items()}

    def instantiate_contract(self, fields: Mapping[str, str]) -> tuple[Literal, ...]:
        return tuple(
            Literal(
                atom.predicate,
                tuple(_substitute(term, fields) for term in atom.terms),
                atom.positive,
            )
            for atom in self.local_contract
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "arguments": dict(self.arguments),
            "local_contract": [atom.as_dict() for atom in self.local_contract],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ToolProgram":
        return cls(
            tool_name=str(value["tool_name"]),
            arguments=dict(value.get("arguments", {})),
            local_contract=tuple(Literal.from_dict(atom) for atom in value.get("local_contract", [])),
        )
