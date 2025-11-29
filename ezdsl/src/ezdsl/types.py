"""
Type system domain for runtime type representation.

This module defines the runtime type representation system used for schema
generation and type extraction. It includes primitive types, type definitions,
and utilities for working with generic types.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import dataclass_transform, get_args, get_origin, Any, ClassVar

# =============================================================================
# Primitives
# =============================================================================

PRIMITIVES: frozenset[type] = frozenset({float, int, str, bool, type(None)})

# =============================================================================
# Type Definitions
# =============================================================================

@dataclass_transform(frozen_default=True)
class TypeDef:
    """Base for type definitions."""

    _tag: ClassVar[str]
    _registry: ClassVar[dict[str, type[TypeDef]]] = {}

    def __init_subclass__(cls, tag: str | None = None, **kwargs):
        super().__init_subclass__(**kwargs)
        if not cls.__dict__.get("__annotations__"):
            return
        dataclass(frozen=True)(cls)
        cls._tag = tag or cls.__name__.lower().removesuffix("type")
        TypeDef._registry[cls._tag] = cls


class PrimitiveType(TypeDef, tag="primitive"):
    primitive: type


class NodeType(TypeDef, tag="node"):
    returns: TypeDef


class RefType(TypeDef, tag="ref"):
    target: TypeDef


class UnionType(TypeDef, tag="union"):
    options: tuple[TypeDef, ...]


class GenericType(TypeDef, tag="generic"):
    """Represents a generic/parameterized type."""
    name: str
    origin: str  # Name of the generic origin (e.g., "list", "dict", "Ref")
    args: tuple[TypeDef, ...]


class TypeVarType(TypeDef, tag="typevar"):
    """Represents a type variable with optional bounds."""
    name: str
    bounds: tuple[TypeDef, ...] | None = None


# =============================================================================
# Type Parameter Substitution
# =============================================================================

def _substitute_type_params(type_expr: Any, substitutions: dict[Any, Any]) -> Any:
    """
    Recursively substitute type parameters in a type expression.

    Args:
        type_expr: The type expression to substitute in
        substitutions: Mapping from type parameters to their concrete types

    Returns:
        The type expression with parameters substituted
    """
    import types

    # If this is a type parameter, substitute it
    if type_expr in substitutions:
        return substitutions[type_expr]

    # Get origin and args for generic types
    origin = get_origin(type_expr)
    args = get_args(type_expr)

    # If no origin, this is a simple type - return as-is
    if origin is None:
        return type_expr

    # If there are no args, return as-is
    if not args:
        return type_expr

    # Recursively substitute in the arguments
    new_args = tuple(_substitute_type_params(arg, substitutions) for arg in args)

    # Handle UnionType (created by | operator) specially
    if isinstance(type_expr, types.UnionType):
        # Reconstruct union using | operator
        result = new_args[0]
        for arg in new_args[1:]:
            result = result | arg
        return result

    # Reconstruct the type with substituted arguments
    return origin[new_args]
