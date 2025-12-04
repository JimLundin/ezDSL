"""AST container with flat node storage and reference resolution."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from nanodsl.nodes import Node, Ref
from nanodsl.serialization import to_dict, from_dict


@dataclass
class AST:
    """Flat AST with nodes stored by ID."""

    root: str
    nodes: dict[str, Node[Any]]

    def resolve[X](self, ref: Ref[X]) -> X:
        """Resolve a reference to its node."""
        return self.nodes[ref.id]

    def to_dict(self) -> dict:
        return {
            "root": self.root,
            "nodes": {k: to_dict(v) for k, v in self.nodes.items()},
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> AST:
        return cls(data["root"], {k: from_dict(v) for k, v in data["nodes"].items()})

    @classmethod
    def from_json(cls, s: str) -> AST:
        return cls.from_dict(json.loads(s))
