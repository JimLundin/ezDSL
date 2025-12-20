"""Core AST node infrastructure with automatic registration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, dataclass_transform


@dataclass(frozen=True)
class Ref[X]:
    """Reference to X by ID."""

    id: str


@dataclass(frozen=True)
@dataclass_transform(frozen_default=True)
class Node[T]:
    """Base for AST nodes. T is return type."""

    tag: ClassVar[str]
    signature: ClassVar[dict[str, Any]]
    registry: ClassVar[dict[str, type[Node[Any]]]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Register node subclass with automatic tag derivation."""
        dataclass(frozen=True)(cls)

        cls.signature = kwargs
        cls.tag = ".".join(map(str, kwargs.values())) if kwargs else cls.__name__

        if (existing := Node.registry.get(cls.tag)) and existing is not cls:
            msg = (
                f"Tag '{cls.tag}' already registered to {existing}. "
                "Choose a different tag."
            )
            raise ValueError(msg)

        Node.registry[cls.tag] = cls


type NodeRef[T] = Ref[Node[T]]
type Child[T] = Node[T] | Ref[Node[T]]
