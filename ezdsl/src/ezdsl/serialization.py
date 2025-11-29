"""
Serialization domain for converting between Python objects and serialized formats.

This module handles conversion between Python objects (Node, Ref, TypeDef) and
their serialized representations (dict/JSON).
"""

from __future__ import annotations

from dataclasses import fields as dc_fields
from typing import get_origin, Any
import json

from ezdsl.nodes import Node, Ref
from ezdsl.types import TypeDef, PRIMITIVES

# =============================================================================
# Serialization
# =============================================================================

def to_dict(obj: Node | Ref | TypeDef) -> dict:
    """Serialize to dict."""
    if isinstance(obj, Ref):
        return {"tag": "ref", "id": obj.id}

    tag = getattr(type(obj), "_tag", None)
    if tag is None:
        raise ValueError(f"No tag for {type(obj)}")

    result = {"tag": tag}
    for field in dc_fields(obj):
        result[field.name] = _serialize_value(getattr(obj, field.name))
    return result


def _serialize_value(value: Any) -> Any:
    if isinstance(value, (Node, Ref, TypeDef)):
        return to_dict(value)
    if isinstance(value, tuple):
        return [_serialize_value(v) for v in value]
    if isinstance(value, list):
        return [_serialize_value(v) for v in value]
    if isinstance(value, type):
        return "none" if value is type(None) else value.__name__
    return value


def from_dict(data: dict, registry: dict[str, type] = None) -> Node | Ref | TypeDef:
    """Deserialize from dict."""
    tag = data["tag"]

    if tag == "ref":
        return Ref(id=data["id"])

    # Try Node registry first, then TypeDef
    registry = registry or Node._registry
    cls = registry.get(tag) or TypeDef._registry.get(tag)
    if cls is None:
        raise ValueError(f"Unknown tag: {tag}")

    kwargs = {}
    for field in dc_fields(cls):
        raw = data.get(field.name)
        kwargs[field.name] = _deserialize_value(raw, field.type)
    return cls(**kwargs)


def _deserialize_value(value: Any, hint: Any) -> Any:
    if isinstance(value, dict) and "tag" in value:
        return from_dict(value)
    if isinstance(value, list):
        return tuple(_deserialize_value(v, hint) for v in value)
    if hint is type or (get_origin(hint) is type):
        # Reconstruct type from name
        for p in PRIMITIVES:
            if ("none" if p is type(None) else p.__name__) == value:
                return p
        raise ValueError(f"Unknown primitive: {value}")
    return value


def to_json(obj: Node | Ref | TypeDef) -> str:
    return json.dumps(to_dict(obj), indent=2)


def from_json(s: str) -> Node | Ref | TypeDef:
    return from_dict(json.loads(s))
