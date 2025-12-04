"""Serialization functions for Node, Ref, and TypeDef."""

from __future__ import annotations

from typing import Any
import json

from nanodsl.nodes import Node, Ref
from nanodsl.types import TypeDef
from nanodsl.adapters import JSONAdapter

_adapter = JSONAdapter()


def to_dict(obj: Node[Any] | Ref[Any] | TypeDef) -> dict:
    """Serialize to dict."""
    if isinstance(obj, Ref):
        return {"tag": "ref", "id": obj.id}
    if isinstance(obj, Node):
        return _adapter.serialize_node(obj)
    if isinstance(obj, TypeDef):
        return _adapter.serialize_typedef(obj)
    raise ValueError(f"Cannot serialize {type(obj)}")


def from_dict(data: dict) -> Node | Ref | TypeDef:
    """Deserialize from dict."""
    tag = data["tag"]
    if tag == "ref":
        return Ref[Any](id=data["id"])
    if tag in Node._registry:
        return _adapter.deserialize_node(data)
    if tag in TypeDef._registry:
        return _adapter.deserialize_typedef(data)
    raise ValueError(f"Unknown tag: {tag}")


def to_json(obj: Node | Ref | TypeDef) -> str:
    """Serialize to JSON string."""
    return json.dumps(to_dict(obj), indent=2)


def from_json(s: str) -> Node | Ref | TypeDef:
    """Deserialize from JSON string."""
    return from_dict(json.loads(s))
