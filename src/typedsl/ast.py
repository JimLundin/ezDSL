"""AST container with flat node storage and reference resolution."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from typedsl.nodes import Node
from typedsl.serialization import from_dict, to_dict

if TYPE_CHECKING:
    from collections.abc import Mapping

    from typedsl.nodes import Ref


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


class Interpreter[Ctx, R](ABC):
    """Base class for AST interpreters.

    Provides AST access, context, and reference resolution.
    Subclass and implement `eval` with your preferred signature.

    Ctx: Type of evaluation context (use None if no context needed)
    R: Return type of run()
    """

    def __init__(self, ast: AST, ctx: Ctx) -> None:
        """Initialize the interpreter.

        Args:
            ast: The AST to interpret
            ctx: The evaluation context

        """
        self.ast = ast
        self.ctx = ctx

    def run(self) -> R:
        """Evaluate the AST from its root node."""
        root = self.ast.nodes[self.ast.root]
        return self.eval(root)

    def resolve[X](self, ref: Ref[X]) -> X:
        """Resolve a reference to its target."""
        return self.ast.resolve(ref)

    @abstractmethod
    def eval(self, node: Node[Any]) -> Any:
        """Evaluate a node. Implement with pattern matching.

        Override with your preferred signature:
        - Homogeneous: def eval(self, node: Node[Any]) -> float
        - Typed: def eval[T](self, node: Node[T]) -> T
        """
        ...
