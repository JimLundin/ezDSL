"""
Schema extraction domain for runtime type reflection and schema generation.

This module provides utilities to reflect on Python types and extract runtime
type information for validation, documentation, and tooling support.
"""

from __future__ import annotations

from dataclasses import fields as dc_fields
from typing import get_args, get_origin, get_type_hints, Any, Union, TypeAliasType, TypeVar, ParamSpec
import types
import sys

from ezdsl.nodes import Node, Ref
from ezdsl.types import (
    TypeDef,
    PrimitiveType,
    NodeType,
    RefType,
    UnionType,
    GenericType,
    TypeVarType,
    TypeParamKind,
    Variance,
    PRIMITIVES,
    _substitute_type_params,
)

# Check if TypeVarTuple is available (Python 3.11+)
if sys.version_info >= (3, 11):
    from typing import TypeVarTuple
else:
    TypeVarTuple = None
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
        return _extract_typevar(py_type)

    # Handle ParamSpec (PEP 612)
    if isinstance(py_type, ParamSpec):
        return TypeVarType(
            name=py_type.__name__,
            kind=TypeParamKind.PARAMSPEC,
            variance=Variance.INVARIANT,
        )

    # Handle TypeVarTuple (PEP 646, Python 3.11+)
    if TypeVarTuple is not None and isinstance(py_type, TypeVarTuple):
        return TypeVarType(
            name=py_type.__name__,
            kind=TypeParamKind.TYPEVARTUPLE,
            variance=Variance.INVARIANT,
        )

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
        extracted_args = tuple(extract_type(a) for a in args)

        # Try to extract the origin as a TypeDef
        # For built-in types, we'll use a primitive representation
        origin_typedef = _extract_generic_origin(origin)

        return GenericType(
            name=f"{origin_name}[{', '.join(str(a) for a in args)}]",
            origin=origin_typedef,
            args=extracted_args
        )

    raise ValueError(f"Cannot extract type from: {py_type}")


def _extract_typevar(typevar: TypeVar) -> TypeVarType:
    """Extract TypeVar metadata into TypeVarType."""
    # Determine variance
    variance = Variance.INVARIANT
    if getattr(typevar, "__covariant__", False):
        variance = Variance.COVARIANT
    elif getattr(typevar, "__contravariant__", False):
        variance = Variance.CONTRAVARIANT

    # Extract constraints (mutually exclusive with bounds)
    constraints_tuple = getattr(typevar, "__constraints__", ())
    constraints = None
    if constraints_tuple:
        constraints = tuple(extract_type(c) for c in constraints_tuple)

    # Extract bound (mutually exclusive with constraints)
    bound = getattr(typevar, "__bound__", None)
    bounds = None
    if bound is not None:
        bounds = (extract_type(bound),)

    # Extract default (Python 3.13+)
    default = None
    if hasattr(typevar, "__default__"):
        default_value = getattr(typevar, "__default__")
        # Check if default is actually set (not the sentinel)
        try:
            from typing import NoDefault
            if default_value is not NoDefault:
                default = extract_type(default_value)
        except ImportError:
            # Python < 3.13, no default support
            pass

    return TypeVarType(
        name=typevar.__name__,
        kind=TypeParamKind.TYPEVAR,
        variance=variance,
        bounds=bounds,
        constraints=constraints,
        default=default,
    )


def _extract_generic_origin(origin: Any) -> TypeDef:
    """
    Extract the origin type of a generic as a TypeDef.

    For built-in types like list, dict, we create a special representation.
    For custom types, we try to extract them properly.
    """
    # Handle primitives that might be generic origins
    if origin in PRIMITIVES:
        return PrimitiveType(origin)

    # Handle Node types
    if isinstance(origin, type) and issubclass(origin, Node):
        return NodeType(_extract_node_returns(origin))

    # Handle built-in generic types (list, dict, set, etc.)
    # We'll represent them as a special primitive-like type
    if hasattr(origin, "__name__"):
        # For now, we'll create a simple type representation
        # In a more complete system, we might want a separate BuiltinGenericType
        # But for simplicity, we'll use a PrimitiveType with the origin type
        return PrimitiveType(origin)

    # Fallback: represent as type(None) - this shouldn't happen often
    return PrimitiveType(type(None))


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
