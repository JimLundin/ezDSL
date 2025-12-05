"""AST container with flat node storage and reference resolution."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from nanodsl.nodes import Node
from nanodsl.serialization import from_dict, to_dict

if TYPE_CHECKING:
    from collections.abc import Mapping

    from nanodsl.nodes import Ref


@dataclass
class AST:
    """Flat AST with nodes stored by ID."""

    root: str
    nodes: Mapping[str, Node[Any]]

    def resolve[X](self, ref: Ref[X]) -> X:
        """Resolve a reference to its node.

        Args:
            ref: Reference to resolve

        Returns:
            The node referenced by the given ref

        Raises:
            KeyError: If the referenced node ID is not found in the AST

        """
        if ref.id not in self.nodes:
            available = list(self.nodes.keys())
            msg = f"Node '{ref.id}' not found in AST. Available node IDs: {available}"
            raise KeyError(msg)
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
        """Deserialize AST from dictionary.

        Args:
            data: Dictionary containing 'root' and 'nodes' keys

        Returns:
            Deserialized AST instance

        Raises:
            KeyError: If required keys ('root' or 'nodes') are missing
            ValueError: If node deserialization fails

        """
        if "root" not in data:
            msg = "Missing required key 'root' in AST data"
            raise KeyError(msg)
        if "nodes" not in data:
            msg = "Missing required key 'nodes' in AST data"
            raise KeyError(msg)

        nodes = {k: cast(Node[Any], from_dict(v)) for k, v in data["nodes"].items()}
        return cls(data["root"], nodes)

    @classmethod
    def from_json(cls, s: str) -> AST:
        return cls.from_dict(json.loads(s))
