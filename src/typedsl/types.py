"""Runtime type representation system for schema generation."""

from __future__ import annotations

import types
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    dataclass_transform,
    get_args,
    get_origin,
)

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class ExternalTypeRecord[T]:
    """Record for external type registration."""

    python_type: type[T]
    module: str  # Full module path, e.g., "pandas.core.frame"
    name: str  # Class name, e.g., "DataFrame"
    encode: Callable[[T], dict[str, Any]]
    decode: Callable[[dict[str, Any]], T]


@dataclass(frozen=True)
@dataclass_transform(frozen_default=True)
class TypeDef:
    """Base for type definitions."""

    _tag: ClassVar[str]
    registry: ClassVar[dict[str, type[TypeDef]]] = {}
    _external_types: ClassVar[dict[type, ExternalTypeRecord[Any]]] = {}

    def __init_subclass__(cls, tag: str | None = None):
        dataclass(frozen=True)(cls)
        cls._tag = tag or cls.__name__.lower().removesuffix("type")

        if existing := TypeDef.registry.get(cls._tag):
            if existing is not cls:
                msg = (
                    f"Tag '{cls._tag}' already registered to {existing}. "
                    "Choose a different tag."
                )
                raise ValueError(msg)

        TypeDef.registry[cls._tag] = cls

    @classmethod
    def register[T](
        cls,
        python_type: type[T],
        *,
        encode: Callable[[T], dict[str, Any]],
        decode: Callable[[dict[str, Any]], T],
    ) -> type[T]:
        """Register an external type for serialization. Returns the type unchanged."""
        module = python_type.__module__
        name = python_type.__name__

        if python_type in cls._external_types:
            existing = cls._external_types[python_type]
            # Idempotent if same module/name
            if existing.module == module and existing.name == name:
                return python_type
            msg = (
                f"Type {python_type} already registered as "
                f"{existing.module}.{existing.name}"
            )
            raise ValueError(msg)

        # Create and store record
        record = ExternalTypeRecord(
            python_type=python_type,
            module=module,
            name=name,
            encode=encode,
            decode=decode,
        )
        cls._external_types[python_type] = record
        return python_type

    @classmethod
    def get_registered_type(cls, python_type: type) -> TypeDef | None:
        """Get ExternalType for a registered Python type, or None if not registered."""
        if python_type in cls._external_types:
            record = cls._external_types[python_type]
            return ExternalType(module=record.module, name=record.name)
        return None


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


class ListType(TypeDef, tag="list"):
    """List type: list[int] → ListType(element=IntType())."""

    element: TypeDef


class DictType(TypeDef, tag="dict"):
    """Dict type: dict[str, int] → DictType(key=StrType(), value=IntType())."""

    key: TypeDef
    value: TypeDef


class SetType(TypeDef, tag="set"):
    """Set type: set[int] → SetType(element=IntType())."""

    element: TypeDef


class TupleType(TypeDef, tag="tuple"):
    """Fixed-length heterogeneous tuple: tuple[int, str] → TupleType(elements=(...))."""

    elements: tuple[TypeDef, ...]


class LiteralType(TypeDef, tag="literal"):
    """Literal enumeration: Literal["a", "b"] → LiteralType(values=("a", "b"))."""

    values: tuple[str | int | bool, ...]


class NodeType(TypeDef, tag="node"):
    """Node type: Node[float] → NodeType(returns=FloatType())."""

    returns: TypeDef


class RefType(TypeDef, tag="ref"):
    """Reference type: Ref[Node[int]] → RefType(target=NodeType(...))."""

    target: TypeDef


class UnionType(TypeDef, tag="union"):
    """Union type: int | str → UnionType(options=(IntType(), StrType()))."""

    options: tuple[TypeDef, ...]


class TypeParameter(TypeDef, tag="typeparam"):
    """Type parameter declaration (e.g., T in class Foo[T])."""

    name: str
    bound: TypeDef | None = None


class TypeParameterRef(TypeDef, tag="typeparamref"):
    """Reference to a type parameter (e.g., T used in a field annotation)."""

    name: str


class ExternalType(TypeDef, tag="external"):
    """Registered external type, identified by module path and class name."""

    module: str
    name: str


def _substitute_type_params(type_expr: Any, substitutions: dict[Any, Any]) -> Any:
    """Recursively substitute type parameters in a type expression."""
    if type_expr in substitutions:
        return substitutions[type_expr]

    origin = get_origin(type_expr)
    args = get_args(type_expr)

    if origin is None or not args:
        return type_expr

    new_args = tuple(_substitute_type_params(arg, substitutions) for arg in args)

    # UnionType (| operator) needs special reconstruction
    if isinstance(type_expr, types.UnionType):
        result = new_args[0]
        for arg in new_args[1:]:
            result = result | arg
        return result

    return origin[new_args]
