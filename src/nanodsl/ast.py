"""AST container with flat node storage and reference resolution."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from nanodsl.serialization import from_dict, to_dict

if TYPE_CHECKING:
    from nanodsl.nodes import Node, Ref


@dataclass
class AST:
    """Flat AST with nodes stored by ID."""

    root: str
    nodes: dict[str, Node[Any]]

    def resolve[X](self, ref: Ref[X]) -> X:
        """Resolve a reference to its node."""
        return cast(X, self.nodes[ref.id])

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "nodes": {k: to_dict(v) for k, v in self.nodes.items()},
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AST:
        nodes = {k: from_dict(v) for k, v in data["nodes"].items()}
        return cls(data["root"], nodes)  # type: ignore[arg-type]

    @classmethod
    def from_json(cls, s: str) -> AST:
        return cls.from_dict(json.loads(s))
