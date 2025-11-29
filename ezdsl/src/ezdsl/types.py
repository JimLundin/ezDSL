"""
Type system domain for runtime type representation.

This module defines the runtime type representation system used for schema
generation and type extraction. It includes primitive types, type definitions,
and utilities for working with generic types.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from typing import dataclass_transform, get_args, get_origin, Any, ClassVar
from enum import Enum

# =============================================================================
# Primitives
# =============================================================================

PRIMITIVES: frozenset[type] = frozenset({float, int, str, bool, type(None)})

# =============================================================================
# Type Parameter Metadata
# =============================================================================

class TypeParamKind(Enum):
    """Kind of type parameter (PEP 695)."""
    TYPEVAR = "typevar"           # Regular type variable (T)
    PARAMSPEC = "paramspec"       # Parameter specification (**P)
    TYPEVARTUPLE = "typevartuple" # Type variable tuple (*Ts)


class Variance(Enum):
    """Variance of a type parameter."""
    INVARIANT = "invariant"       # T (default)
    COVARIANT = "covariant"       # T_co (can be subtype)
    CONTRAVARIANT = "contravariant"  # T_contra (can be supertype)

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
    """
    Represents a parameterized/applied generic type.

    Examples: list[int], dict[str, float], Node[int]
    This is a concrete application of a generic type with specific type arguments.
    """
    name: str  # Full name like "list[int]"
    origin: TypeDef  # The generic origin type
    args: tuple[TypeDef, ...]  # Type arguments


class TypeVarType(TypeDef, tag="typevar"):
    """
    Represents a type variable or type parameter.

    Handles both old-style TypeVar and PEP 695 type parameters.

    Examples:
        - TypeVar('T')
        - TypeVar('T', bound=int)
        - TypeVar('T', int, str)  # constraints
        - class Foo[T]: ...  # PEP 695 syntax
    """
    name: str
    kind: TypeParamKind = TypeParamKind.TYPEVAR
    variance: Variance = Variance.INVARIANT
    bounds: tuple[TypeDef, ...] | None = None  # Upper bounds
    constraints: tuple[TypeDef, ...] | None = None  # Type constraints
    default: TypeDef | None = None  # Default type (Python 3.13+)


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
