"""Serialization functions for Node, Ref, and TypeDef."""

from __future__ import annotations

import json
from typing import Any

from typedsl.adapters import JSONAdapter
from typedsl.nodes import Node, Ref
from typedsl.types import TypeDef

_adapter = JSONAdapter()


def to_dict(obj: Node[Any] | Ref[Any] | TypeDef) -> dict[str, Any]:
    """Serialize object to dictionary.

    Args:
        obj: Node, Ref, or TypeDef to serialize

    Returns:
        Dictionary representation of the object

    Raises:
        ValueError: If object type cannot be serialized

    """
    if isinstance(obj, Ref):
        return {"tag": "ref", "id": obj.id}
    if isinstance(obj, Node):
        return _adapter.serialize_node(obj)
    if isinstance(obj, TypeDef):
        return _adapter.serialize_typedef(obj)
    msg = f"Cannot serialize object of type {type(obj).__name__}"
    raise ValueError(msg)


def from_dict(data: dict[str, Any]) -> Node[Any] | Ref[Any] | TypeDef:
    """Deserialize object from dictionary.

    Args:
        data: Dictionary containing serialized object with 'tag' field

    Returns:
        Deserialized Node, Ref, or TypeDef instance

    Raises:
        KeyError: If required 'tag' field is missing
        ValueError: If tag is not recognized

    """
    if "tag" not in data:
        msg = "Missing required 'tag' field in data"
        raise KeyError(msg)

    tag = data["tag"]

    if tag == "ref":
        if "id" not in data:
            msg = "Missing required 'id' field for ref"
            raise KeyError(msg)
        return Ref[Any](id=data["id"])

    if tag in Node.registry:
        return _adapter.deserialize_node(data)

    if tag in TypeDef.registry:
        return _adapter.deserialize_typedef(data)

    # Provide helpful error with available tags
    available_node_tags = list(Node.registry.keys())
    available_typedef_tags = list(TypeDef.registry.keys())
    msg = (
        f"Unknown tag '{tag}'. "
        f"Available node tags: {available_node_tags[:10]}{'...' if len(available_node_tags) > 10 else ''}. "
        f"Available typedef tags: {available_typedef_tags[:10]}{'...' if len(available_typedef_tags) > 10 else ''}"
    )
    raise ValueError(msg)


def to_json(obj: Node[Any] | Ref[Any] | TypeDef) -> str:
    """Serialize object to JSON string.

    Args:
        obj: Node, Ref, or TypeDef to serialize

    Returns:
        JSON string representation (formatted with 2-space indent)

    Raises:
        ValueError: If object type cannot be serialized

    """
    return json.dumps(to_dict(obj), indent=2)


def from_json(s: str) -> Node[Any] | Ref[Any] | TypeDef:
    """Deserialize object from JSON string.

    Args:
        s: JSON string containing serialized object

    Returns:
        Deserialized Node, Ref, or TypeDef instance

    Raises:
        json.JSONDecodeError: If string is not valid JSON
        KeyError: If required fields are missing
        ValueError: If tag is not recognized

    """
    return from_dict(json.loads(s))
