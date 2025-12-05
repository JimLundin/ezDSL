"""Core AST node infrastructure with automatic registration."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, ClassVar, dataclass_transform

# Tag pattern: must start with lowercase letter, followed by lowercase letters, digits, hyphens, or underscores
_TAG_PATTERN = re.compile(r"^[a-z][a-z0-9_-]*$")


def _validate_tag(tag: str) -> None:
    """Validate that a tag follows the required format.

    Args:
        tag: Tag string to validate

    Raises:
        ValueError: If tag does not match required format

    """
    if not _TAG_PATTERN.match(tag):
        msg = (
            f"Invalid tag '{tag}'. Tags must start with a lowercase letter "
            "and contain only lowercase letters, digits, hyphens, and underscores."
        )
        raise ValueError(msg)


@dataclass(frozen=True)
class Ref[X]:
    """Reference to X by ID."""

    id: str


@dataclass(frozen=True)
@dataclass_transform(frozen_default=True)
class Node[T]:
    """Base for AST nodes. T is return type."""

    _tag: ClassVar[str]
    registry: ClassVar[dict[str, type[Node[Any]]]] = {}

    def __init_subclass__(cls, tag: str | None = None) -> None:
        dataclass(frozen=True)(cls)
        cls._tag = tag if tag is not None else cls.__name__.lower().removesuffix("node")

        # Validate tag format
        _validate_tag(cls._tag)

        if existing := Node.registry.get(cls._tag):
            if existing is not cls:
                msg = (
                    f"Tag '{cls._tag}' already registered to {existing}. "
                    "Choose a different tag."
                )
                raise ValueError(msg)

        Node.registry[cls._tag] = cls


type NodeRef[T] = Ref[Node[T]]
type Child[T] = Node[T] | Ref[Node[T]]
