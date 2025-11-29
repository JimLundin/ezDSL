"""
Schema extraction domain for runtime type reflection and schema generation.

This module provides utilities to reflect on Python types and extract runtime
type information for validation, documentation, and tooling support.
"""

from __future__ import annotations

from dataclasses import fields as dc_fields
from typing import get_args, get_origin, get_type_hints, Any, Union, TypeAliasType, TypeVar
import types

from ezdsl.nodes import Node, Ref
from ezdsl.types import (
    TypeDef,
    PrimitiveType,
    NodeType,
    RefType,
    UnionType,
    GenericType,
    TypeVarType,
    PRIMITIVES,
    _substitute_type_params,
)
from ezdsl.serialization import to_dict

# =============================================================================
# Schema Extraction
# =============================================================================

def extract_type(py_type: Any) -> TypeDef:
    """Convert Python type annotation to TypeDef."""
    origin = get_origin(py_type)
    args = get_args(py_type)

    # Handle TypeVar (old-style and PEP 695 type parameters)
    if isinstance(py_type, TypeVar):
        bounds = getattr(py_type, "__constraints__", None)
        if bounds:
            return TypeVarType(
                name=py_type.__name__,
                bounds=tuple(extract_type(b) for b in bounds)
            )
        bound = getattr(py_type, "__bound__", None)
        if bound is not None:
            return TypeVarType(
                name=py_type.__name__,
                bounds=(extract_type(bound),)
            )
        return TypeVarType(name=py_type.__name__, bounds=None)

    # PEP 695 type aliases - automatically expand them
    if isinstance(origin, TypeAliasType):
        # Get the type parameters and create substitution mapping
        type_params = getattr(origin, "__type_params__", ())
        if len(type_params) != len(args):
            raise ValueError(
                f"Type alias {origin.__name__} expects {len(type_params)} "
                f"arguments but got {len(args)}"
            )

        # Create substitution mapping: {T: int, U: str, ...}
        substitutions = dict(zip(type_params, args))

        # Get the template and substitute type parameters
        template = origin.__value__
        substituted = _substitute_type_params(template, substitutions)

        # Extract the substituted type
        return extract_type(substituted)

    # Primitives
    if py_type in PRIMITIVES:
        return PrimitiveType(py_type)

    # Node types
    if origin is not None and isinstance(origin, type) and issubclass(origin, Node):
        return NodeType(extract_type(args[0]) if args else PrimitiveType(type(None)))

    if isinstance(py_type, type) and issubclass(py_type, Node):
        return NodeType(_extract_node_returns(py_type))

    # Ref types
    if origin is Ref:
        return RefType(extract_type(args[0]) if args else PrimitiveType(type(None)))

    # Union types (typing.Union)
    if origin is Union:
        return UnionType(tuple(extract_type(a) for a in args))

    # UnionType (types.UnionType, created by | operator in Python 3.10+)
    if isinstance(py_type, types.UnionType):
        return UnionType(tuple(extract_type(a) for a in args))

    # Generic types (fallback for other parameterized types)
    if origin is not None and args:
        origin_name = getattr(origin, "__name__", str(origin))
        return GenericType(
            name=f"{origin_name}[{', '.join(str(a) for a in args)}]",
            origin=origin_name,
            args=tuple(extract_type(a) for a in args)
        )

    raise ValueError(f"Cannot extract type from: {py_type}")


def _extract_node_returns(cls: type[Node]) -> TypeDef:
    for base in getattr(cls, "__orig_bases__", ()):
        origin = get_origin(base)
        if origin is not None and isinstance(origin, type) and issubclass(origin, Node):
            args = get_args(base)
            if args:
                return extract_type(args[0])
    return PrimitiveType(type(None))


def node_schema(cls: type[Node]) -> dict:
    """Get schema for a node class."""
    hints = get_type_hints(cls)
    return {
        "tag": cls._tag,
        "returns": to_dict(_extract_node_returns(cls)),
        "fields": [
            {"name": f.name, "type": to_dict(extract_type(hints[f.name]))}
            for f in dc_fields(cls)
            if not f.name.startswith("_")
        ],
    }


def all_schemas() -> dict:
    """Get all registered node schemas."""
    return {"nodes": {tag: node_schema(cls) for tag, cls in Node._registry.items()}}
