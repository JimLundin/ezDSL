"""
Type system domain for runtime type representation.

This module defines the runtime type representation system used for schema
generation and type extraction. It uses concrete types rather than generic
wrappers to provide clear, self-documenting type definitions.
"""

from __future__ import annotations

import types
from dataclasses import dataclass
from typing import dataclass_transform, get_args, get_origin, Any, ClassVar

# =============================================================================
# Type Definition Base
# =============================================================================

@dataclass_transform(frozen_default=True)
class TypeDef:
    """Base for type definitions."""

    _tag: ClassVar[str]
    _registry: ClassVar[dict[str, type[TypeDef]]] = {}

    def __init_subclass__(cls, tag: str | None = None, **kwargs):
        super().__init_subclass__(**kwargs)
        if "__annotations__" not in cls.__dict__:
            return
        # dataclass returns a modified class - we don't need to reassign in __init_subclass__
        # because it modifies the class in place, but we should still call it
        dataclass(frozen=True)(cls)
        cls._tag = tag or cls.__name__.lower().removesuffix("type")
        TypeDef._registry[cls._tag] = cls


# =============================================================================
# Primitive Types (Concrete)
# =============================================================================

class IntType(TypeDef, tag="int"):
    """Integer type."""
    __annotations__ = {}  # Trigger dataclass conversion


class FloatType(TypeDef, tag="float"):
    """Floating point type."""
    __annotations__ = {}  # Trigger dataclass conversion


class StrType(TypeDef, tag="str"):
    """String type."""
    __annotations__ = {}  # Trigger dataclass conversion


class BoolType(TypeDef, tag="bool"):
    """Boolean type."""
    __annotations__ = {}  # Trigger dataclass conversion


class NoneType(TypeDef, tag="none"):
    """None/null type."""
    __annotations__ = {}  # Trigger dataclass conversion


# =============================================================================
# Container Types (Concrete)
# =============================================================================

class ListType(TypeDef, tag="list"):
    """
    List type with element type.

    Example: list[int] → ListType(element=IntType())
    """
    element: TypeDef


class DictType(TypeDef, tag="dict"):
    """
    Dictionary type with key and value types.

    Example: dict[str, int] → DictType(key=StrType(), value=IntType())
    """
    key: TypeDef
    value: TypeDef


# =============================================================================
# Domain Types
# =============================================================================

class NodeType(TypeDef, tag="node"):
    """
    AST Node type with return type.

    Example: Node[float] → NodeType(returns=FloatType())
    """
    returns: TypeDef


class RefType(TypeDef, tag="ref"):
    """
    Reference type pointing to another type.

    Example: Ref[Node[int]] → RefType(target=NodeType(returns=IntType()))
    """
    target: TypeDef


class UnionType(TypeDef, tag="union"):
    """
    Union of multiple types.

    Example: int | str → UnionType(options=(IntType(), StrType()))
    """
    options: tuple[TypeDef, ...]


class TypeParameter(TypeDef, tag="param"):
    """
    Type parameter in PEP 695 syntax.

    Type parameters are the placeholders in generic definitions that get
    substituted with concrete types when the generic is used.

    Examples:
        - class Foo[T]: ...         # T is an unbounded type parameter
        - class Foo[T: int]: ...    # T is bounded (must be int or subtype)
        - type Pair[T] = tuple[T, T]  # T is a type parameter in the alias
    """
    name: str
    bound: TypeDef | None = None  # Upper bound constraint (e.g., T: int)


# =============================================================================
# Custom Type Registry
# =============================================================================

# Global registry for user-defined types
# Maps Python marker classes to their TypeDef representations
_CUSTOM_TYPE_REGISTRY: dict[type, type[TypeDef]] = {}


def register_custom_type(python_type: type, typedef: type[TypeDef]) -> None:
    """
    Register a custom Python type to TypeDef mapping.

    This allows DSL users to define custom types (like DataFrame, Matrix, etc.)
    that can be used in type hints while maintaining IDE support.

    Args:
        python_type: The Python class to use as a type marker (e.g., DataFrame)
        typedef: The TypeDef subclass for serialization (e.g., DataFrameType)

    Example:
        >>> class DataFrame:
        ...     '''User-defined DataFrame type.'''
        ...     pass
        >>>
        >>> class DataFrameType(TypeDef, tag="dataframe"):
        ...     __annotations__ = {}
        >>>
        >>> register_custom_type(DataFrame, DataFrameType)
        >>>
        >>> # Now you can use it in your DSL:
        >>> class FetchData(Node[DataFrame], tag="fetch"):
        ...     query: str
    """
    _CUSTOM_TYPE_REGISTRY[python_type] = typedef


def get_custom_type(python_type: type) -> type[TypeDef] | None:
    """
    Get the registered TypeDef for a Python type.

    Returns None if the type is not registered.
    """
    return _CUSTOM_TYPE_REGISTRY.get(python_type)


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
