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
    _namespace: ClassVar[str]
    _registry: ClassVar[dict[str, type[TypeDef]]] = {}
    _custom_types: ClassVar[dict[type, type[TypeDef]]] = {}  # Maps Python types to TypeDef classes

    def __init_subclass__(cls, tag: str | None = None, namespace: str | None = None, **kwargs):
        super().__init_subclass__(**kwargs)

        # Always convert to frozen dataclass
        dataclass(frozen=True)(cls)

        # Store namespace and base tag
        cls._namespace = namespace or ""
        base_tag = tag or cls.__name__.lower().removesuffix("type")

        # Create full namespaced tag
        cls._tag = f"{namespace}.{base_tag}" if namespace else base_tag

        # Check for collisions
        if cls._tag in TypeDef._registry:
            existing = TypeDef._registry[cls._tag]
            if existing is not cls:
                raise ValueError(
                    f"Tag '{cls._tag}' already registered to {existing}. "
                    f"Choose a different tag or namespace."
                )

        TypeDef._registry[cls._tag] = cls

    @classmethod
    def register(
        cls,
        python_type: type | None = None,
        *,
        tag: str | None = None,
        namespace: str = "custom"
    ) -> type[TypeDef] | Any:
        """
        Register a custom type with the type system.

        This method can be used either as a decorator or as a regular function call
        to register existing types (like pandas DataFrame, numpy ndarray, etc.) with
        the DSL type system.

        Args:
            python_type: The Python class to register. If None, returns a decorator.
            tag: Optional tag name. Defaults to lowercase class name.
            namespace: Namespace for the type. Defaults to "custom".

        Returns:
            If used as decorator: returns the original class unchanged
            If used as function: returns the created TypeDef subclass

        Examples:
            # Register an existing type (e.g., pandas DataFrame)
            >>> import pandas as pd
            >>> TypeDef.register(pd.DataFrame)  # tag="custom.dataframe"
            >>> TypeDef.register(pd.DataFrame, tag="df")  # tag="custom.df"

            # Use as decorator for marker classes
            >>> @TypeDef.register
            ... class GraphicsContext:
            ...     '''Marker for graphics context type.'''
            ...     pass

            # Use as decorator with custom tag
            >>> @TypeDef.register(tag="matrix")
            ... class Matrix:
            ...     pass
        """
        def _create_typedef(py_type: type) -> type[TypeDef]:
            """Create and register a TypeDef for the given Python type."""
            # Determine the tag
            base_tag = tag or py_type.__name__.lower()

            # Check if already registered
            if py_type in cls._custom_types:
                existing = cls._custom_types[py_type]
                full_tag = f"{namespace}.{base_tag}" if namespace else base_tag
                # If same tag, it's idempotent
                if existing._tag == full_tag:
                    return existing
                raise ValueError(
                    f"Type {py_type} already registered with tag '{existing._tag}'. "
                    f"Cannot re-register with tag '{full_tag}'."
                )

            # Create TypeDef subclass dynamically
            typedef_name = f"{py_type.__name__}Type"

            # Build class dict
            class_dict = {
                "__module__": py_type.__module__,
                "__doc__": f"Custom type definition for {py_type.__name__}.",
            }

            # Create the class - type() will call __init_subclass__ with our kwargs
            typedef_cls = type(typedef_name, (TypeDef,), class_dict, tag=base_tag, namespace=namespace)

            # Register the mapping
            cls._custom_types[py_type] = typedef_cls

            return typedef_cls

        # If called without arguments as a decorator: @TypeDef.register
        if python_type is not None:
            typedef = _create_typedef(python_type)
            # When used as decorator, return the original class unchanged
            # This allows the marker class to still be used normally
            return python_type

        # If called with arguments: @TypeDef.register(tag="foo")
        # Return a decorator that will be applied to the class
        def decorator(py_type: type) -> type:
            _create_typedef(py_type)
            return py_type

        return decorator

    @classmethod
    def get_registered_type(cls, python_type: type) -> type[TypeDef] | None:
        """
        Get the registered TypeDef for a Python type.

        Args:
            python_type: The Python type to look up

        Returns:
            The TypeDef class registered for this type, or None if not registered
        """
        return cls._custom_types.get(python_type)


# =============================================================================
# Primitive Types (Concrete)
# =============================================================================

class IntType(TypeDef, tag="int", namespace="std"):
    """Integer type."""
    pass


class FloatType(TypeDef, tag="float", namespace="std"):
    """Floating point type."""
    pass


class StrType(TypeDef, tag="str", namespace="std"):
    """String type."""
    pass


class BoolType(TypeDef, tag="bool", namespace="std"):
    """Boolean type."""
    pass


class NoneType(TypeDef, tag="none", namespace="std"):
    """None/null type."""
    pass


# =============================================================================
# Container Types (Concrete)
# =============================================================================

class ListType(TypeDef, tag="list", namespace="std"):
    """
    List type with element type.

    Example: list[int] → ListType(element=IntType())
    """
    element: TypeDef


class DictType(TypeDef, tag="dict", namespace="std"):
    """
    Dictionary type with key and value types.

    Example: dict[str, int] → DictType(key=StrType(), value=IntType())
    """
    key: TypeDef
    value: TypeDef


# =============================================================================
# Domain Types
# =============================================================================

class NodeType(TypeDef, tag="node", namespace="std"):
    """
    AST Node type with return type.

    Example: Node[float] → NodeType(returns=FloatType())
    """
    returns: TypeDef


class RefType(TypeDef, tag="ref", namespace="std"):
    """
    Reference type pointing to another type.

    Example: Ref[Node[int]] → RefType(target=NodeType(returns=IntType()))
    """
    target: TypeDef


class UnionType(TypeDef, tag="union", namespace="std"):
    """
    Union of multiple types.

    Example: int | str → UnionType(options=(IntType(), StrType()))
    """
    options: tuple[TypeDef, ...]


class TypeParameter(TypeDef, tag="param", namespace="std"):
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
