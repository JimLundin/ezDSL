"""
Type system domain for runtime type representation.

This module defines the runtime type representation system used for schema
generation and type extraction. It uses concrete types rather than generic
wrappers to provide clear, self-documenting type definitions.
"""

from __future__ import annotations

import types
from dataclasses import dataclass
from typing import dataclass_transform, get_args, get_origin, Any, ClassVar, Callable

# =============================================================================
# Type Registration Records
# =============================================================================


@dataclass(frozen=True)
class ExternalTypeRecord:
    """Record for external type registration (function style)."""

    python_type: type
    module: str  # Full module path, e.g., "pandas.core.frame"
    name: str  # Class name, e.g., "DataFrame"
    tag: str
    encode: Callable[[Any], dict]
    decode: Callable[[dict], Any]


@dataclass(frozen=True)
class CustomTypeRecord:
    """Record for custom type registration (decorator style)."""

    python_type: type
    tag: str
    # encode/decode are methods on the class, not stored here


# =============================================================================
# Type Definition Base
# =============================================================================


@dataclass_transform(frozen_default=True)
class TypeDef:
    """Base for type definitions."""

    _tag: ClassVar[str]
    _registry: ClassVar[dict[str, type[TypeDef]]] = {}
    _external_types: ClassVar[dict[type, ExternalTypeRecord]] = {}
    _custom_types: ClassVar[dict[type, CustomTypeRecord]] = {}

    def __init_subclass__(cls, tag: str | None = None):
        # Always convert to frozen dataclass
        dataclass(frozen=True)(cls)

        # Determine tag
        cls._tag = tag or cls.__name__.lower().removesuffix("type")

        # Check for collisions
        if existing := TypeDef._registry.get(cls._tag):
            if existing is not cls:
                raise ValueError(
                    f"Tag '{cls._tag}' already registered to {existing}. "
                    f"Choose a different tag."
                )

        TypeDef._registry[cls._tag] = cls

    @classmethod
    def register(
        cls,
        python_type: type | None = None,
        *,
        tag: str | None = None,
        encode: Callable[[Any], dict] | None = None,
        decode: Callable[[dict], Any] | None = None,
    ) -> type | Callable[[type], type]:
        """
        Register a type with the type system.

        Two registration styles:

        1. External types (function style with encode/decode):
           TypeDef.register(
               pd.DataFrame,
               tag="dataframe",
               encode=lambda df: {"data": df.to_dict()},
               decode=lambda d: pd.DataFrame(d["data"])
           )
           Creates ExternalType(module="pandas.core.frame", name="DataFrame", tag="dataframe")

        2. Custom types (decorator style with encode/decode methods):
           @TypeDef.register(tag="point")
           class Point:
               def encode(self) -> dict: ...
               @classmethod
               def decode(cls, data: dict) -> Self: ...
           Creates CustomType(tag="point")

        Args:
            python_type: The Python class to register. If None, returns a decorator.
            tag: Optional tag name. Defaults to lowercase class name.
            encode: Optional encode function (external types only).
            decode: Optional decode function (external types only).

        Returns:
            If used as decorator: returns the original class unchanged
            If used as function: returns the original class
        """
        # Determine registration style
        is_external = encode is not None or decode is not None

        if is_external:
            # External type registration (function style)
            if python_type is None:
                raise ValueError(
                    "External type registration requires python_type argument"
                )
            if encode is None or decode is None:
                raise ValueError(
                    "External type registration requires both encode and decode functions"
                )

            # Get type info
            type_tag = tag or python_type.__name__.lower()
            module = python_type.__module__
            name = python_type.__name__

            # Check if already registered
            if python_type in cls._external_types:
                existing = cls._external_types[python_type]
                if existing.tag == type_tag:
                    return python_type  # Idempotent
                raise ValueError(
                    f"Type {python_type} already registered as external type "
                    f"with tag '{existing.tag}'"
                )
            if python_type in cls._custom_types:
                raise ValueError(
                    f"Type {python_type} already registered as custom type"
                )

            # Create and store record
            record = ExternalTypeRecord(
                python_type=python_type,
                module=module,
                name=name,
                tag=type_tag,
                encode=encode,
                decode=decode,
            )
            cls._external_types[python_type] = record
            return python_type

        else:
            # Custom type registration (decorator style)
            def _register_custom(py_type: type) -> type:
                type_tag = tag or py_type.__name__.lower()

                # Check if already registered
                if py_type in cls._custom_types:
                    existing = cls._custom_types[py_type]
                    if existing.tag == type_tag:
                        return py_type  # Idempotent
                    raise ValueError(
                        f"Type {py_type} already registered as custom type "
                        f"with tag '{existing.tag}'"
                    )
                if py_type in cls._external_types:
                    raise ValueError(
                        f"Type {py_type} already registered as external type"
                    )

                # Note: encode/decode methods are optional for marker classes
                # They're only required if you actually want to serialize/deserialize
                # instances of this type

                # Create and store record
                record = CustomTypeRecord(python_type=py_type, tag=type_tag)
                cls._custom_types[py_type] = record
                return py_type

            # If python_type provided, register directly
            if python_type is not None:
                return _register_custom(python_type)

            # Otherwise return decorator
            return _register_custom

    @classmethod
    def get_registered_type(cls, python_type: type) -> "TypeDef | None":
        """
        Get the registered TypeDef for a Python type.

        Args:
            python_type: The Python type to look up

        Returns:
            ExternalType or CustomType instance for this type, or None if not registered
        """
        # Check external types first
        if python_type in cls._external_types:
            record = cls._external_types[python_type]
            # Import here to avoid circular dependency
            # ExternalType is defined below in this same file
            return ExternalType(
                module=record.module, name=record.name, tag=record.tag
            )

        # Check custom types
        if python_type in cls._custom_types:
            record = cls._custom_types[python_type]
            # Import here to avoid circular dependency
            # CustomType is defined below in this same file
            return CustomType(tag=record.tag)

        return None


# =============================================================================
# Primitive Types (Concrete)
# =============================================================================


class IntType(TypeDef, tag="int"):
    """Integer type."""


class FloatType(TypeDef, tag="float"):
    """Floating point type."""


class StrType(TypeDef, tag="str"):
    """String type."""


class BoolType(TypeDef, tag="bool"):
    """Boolean type."""


class NoneType(TypeDef, tag="none"):
    """None/null type."""


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


class SetType(TypeDef, tag="set"):
    """
    Set type with element type.

    Example: set[int] → SetType(element=IntType())
    """

    element: TypeDef


class TupleType(TypeDef, tag="tuple"):
    """
    Fixed-length heterogeneous tuple type.

    Unlike list (homogeneous), tuple types have:
    - Fixed length (known at schema time)
    - Heterogeneous element types (each position can have different type)

    Examples:
        tuple[int, str, float] → TupleType(elements=(IntType(), StrType(), FloatType()))
        tuple[str, str, str] → TupleType(elements=(StrType(), StrType(), StrType()))
    """

    elements: tuple[TypeDef, ...]


class LiteralType(TypeDef, tag="literal"):
    """
    Literal type representing enumeration of values.

    Maps Python's Literal[...] type to enumeration schema.

    Examples:
        Literal["red", "green", "blue"] → LiteralType(values=("red", "green", "blue"))
        Literal[1, 2, 3] → LiteralType(values=(1, 2, 3))
        Literal[True, False] → LiteralType(values=(True, False))

    Note: Does not support Python enum.Enum at this stage.
    """

    values: tuple[str | int | bool, ...]


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


class TypeParameter(TypeDef, tag="typeparam"):
    """
    Type parameter declaration in PEP 695 generic definitions.

    Represents the DECLARATION of a type parameter (e.g., in class Foo[T]).

    Examples:
        class Foo[T]: ... → TypeParameter(name="T", bound=None)
        class Foo[T: int | float]: ... → TypeParameter(name="T", bound=UnionType(...))

    This is the definition site of the type parameter.
    """

    name: str
    bound: TypeDef | None = None


class TypeParameterRef(TypeDef, tag="typeparamref"):
    """
    Reference to a type parameter within a type expression.

    Represents a USE of a type parameter (e.g., in field: T).

    Examples:
        In class Foo[T]:
            field: T → TypeParameterRef(name="T")
            field: list[T] → ListType(element=TypeParameterRef(name="T"))

    This is the use site that refers back to the TypeParameter declaration.
    """

    name: str


class ExternalType(TypeDef, tag="external"):
    """
    Reference to an externally registered type.

    Used for third-party types like pandas.DataFrame, polars.DataFrame, etc.
    Stores module and name to avoid collisions between different libraries.

    Examples:
        pd.DataFrame → ExternalType(module="pandas.core.frame", name="DataFrame", tag="pd_dataframe")
        pl.DataFrame → ExternalType(module="polars.dataframe.frame", name="DataFrame", tag="pl_dataframe")
    """

    module: str  # Full module path
    name: str    # Class name
    tag: str     # User-supplied tag


class CustomType(TypeDef, tag="custom"):
    """
    Reference to a user-defined custom type.

    Used for types registered with the decorator style, where encode/decode
    are methods on the class itself.

    Example:
        @TypeDef.register(tag="point")
        class Point: ...
        → CustomType(tag="point")
    """

    tag: str


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
