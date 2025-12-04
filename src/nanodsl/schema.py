"""Schema extraction and type reflection utilities."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import (
    get_args,
    get_origin,
    get_type_hints,
    Any,
    TypeAliasType,
    TypeVar,
    Literal,
)
import types

from nanodsl.nodes import Node, Ref
from nanodsl.types import (
    TypeDef,
    IntType,
    FloatType,
    StrType,
    BoolType,
    NoneType,
    ListType,
    DictType,
    SetType,
    TupleType,
    LiteralType,
    NodeType,
    RefType,
    UnionType,
    TypeParameter,
    TypeParameterRef,
    ExternalType,
    _substitute_type_params,
)


@dataclass(frozen=True)
class FieldSchema:
    """Schema for a node field."""

    name: str
    type: TypeDef


@dataclass(frozen=True)
class NodeSchema:
    """Complete schema for a node class."""

    tag: str
    type_params: tuple[TypeParameter, ...]  # Type parameter declarations
    returns: TypeDef
    fields: tuple[FieldSchema, ...]


def extract_type(py_type: Any) -> TypeDef:
    """Convert Python type annotation to TypeDef."""
    origin = get_origin(py_type)
    args = get_args(py_type)

    if isinstance(py_type, TypeVar):
        bound = getattr(py_type, "__bound__", None)
        return TypeParameter(
            name=py_type.__name__,
            bound=extract_type(bound) if bound is not None else None,
        )

    custom_typedef = TypeDef.get_registered_type(py_type)
    if custom_typedef is not None:
        return custom_typedef

    # Expand PEP 695 type aliases
    if isinstance(origin, TypeAliasType):
        type_params = getattr(origin, "__type_params__", ())
        if len(type_params) != len(args):
            raise ValueError(
                f"Type alias {origin.__name__} expects {len(type_params)} "
                f"arguments but got {len(args)}"
            )
        substitutions = dict(zip(type_params, args))
        substituted = _substitute_type_params(origin.__value__, substitutions)
        return extract_type(substituted)

    if py_type is int:
        return IntType()
    if py_type is float:
        return FloatType()
    if py_type is str:
        return StrType()
    if py_type is bool:
        return BoolType()
    if py_type is type(None):
        return NoneType()

    if origin is list:
        if not args:
            raise ValueError("list type must have an element type")
        return ListType(element=extract_type(args[0]))

    if origin is dict:
        if len(args) != 2:
            raise ValueError("dict type must have key and value types")
        return DictType(key=extract_type(args[0]), value=extract_type(args[1]))

    if origin is set:
        if not args:
            raise ValueError("set type must have an element type")
        return SetType(element=extract_type(args[0]))

    if origin is tuple:
        if not args:
            raise ValueError("tuple type must have element types")
        return TupleType(elements=tuple(extract_type(arg) for arg in args))

    if origin is Literal:
        if not args:
            raise ValueError("Literal type must have values")
        for val in args:
            if not isinstance(val, (str, int, bool)):
                raise ValueError(
                    f"Literal values must be str, int, or bool, got {type(val)}"
                )
        return LiteralType(values=args)

    if origin is not None and isinstance(origin, type) and issubclass(origin, Node):
        return NodeType(extract_type(args[0]) if args else NoneType())

    if isinstance(py_type, type) and issubclass(py_type, Node):
        return NodeType(_extract_node_returns(py_type))

    if origin is Ref:
        return RefType(extract_type(args[0]) if args else NoneType())

    if isinstance(py_type, types.UnionType):
        return UnionType(tuple(extract_type(a) for a in args))

    raise ValueError(f"Cannot extract type from: {py_type}")


def _extract_node_returns(cls: type[Node[Any]]) -> TypeDef:
    """Extract the return type from a Node class definition."""
    for base in getattr(cls, "__orig_bases__", ()):
        origin = get_origin(base)
        if origin is not None and isinstance(origin, type) and issubclass(origin, Node):
            args = get_args(base)
            if args:
                return extract_type(args[0])
    return NoneType()


def node_schema(cls: type[Node[Any]]) -> NodeSchema:
    """Get schema for a node class."""
    hints = get_type_hints(cls)

    type_params: list[TypeParameter] = []
    if hasattr(cls, "__type_params__"):
        for param in cls.__type_params__:
            if isinstance(param, TypeVar):
                bound = getattr(param, "__bound__", None)
                type_params.append(
                    TypeParameter(
                        name=param.__name__,
                        bound=extract_type(bound) if bound is not None else None,
                    )
                )

    fields: list[FieldSchema] = []
    for f in dataclasses.fields(cls):
        if not f.name.startswith("_"):
            fields.append(
                FieldSchema(name=f.name, type=extract_type(hints[f.name]))
            )

    return NodeSchema(
        tag=cls._tag,
        type_params=tuple(type_params),
        returns=_extract_node_returns(cls),
        fields=tuple(fields),
    )


def all_schemas() -> dict[str, NodeSchema]:
    """Get all registered node schemas."""
    return {tag: node_schema(cls) for tag, cls in Node._registry.items()}
