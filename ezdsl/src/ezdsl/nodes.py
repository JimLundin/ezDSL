"""
Node system domain for AST node infrastructure.

This module provides the core AST node infrastructure with automatic registration
and generic type parameters. Nodes are immutable and type-safe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import dataclass_transform, ClassVar

# =============================================================================
# Core Types
# =============================================================================

@dataclass(frozen=True)
class Ref[X]:
    """Reference to X by ID."""
    id: str


type NodeRef[T] = Ref[Node[T]]
type Child[T] = Node[T] | Ref[Node[T]]


@dataclass_transform(frozen_default=True)
class Node[T]:
    """Base for AST nodes. T is return type."""

    _tag: ClassVar[str]
    _registry: ClassVar[dict[str, type[Node]]] = {}

    def __init_subclass__(cls, tag: str | None = None, frozen: bool = True, **kwargs):
        super().__init_subclass__(**kwargs)
        if not cls.__dict__.get("__annotations__"):
            return
        dataclass(frozen=frozen, eq=True, repr=True)(cls)
        cls._tag = tag or cls.__name__.lower()
        Node._registry[cls._tag] = cls
