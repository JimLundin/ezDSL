"""Schema extraction and type reflection utilities."""

from __future__ import annotations

import types
from dataclasses import dataclass, fields
from typing import (
    Any,
    Literal,
    TypeAliasType,
    TypeVar,
    get_args,
    get_origin,
    get_type_hints,
)

from typedsl.nodes import Node, Ref
from typedsl.types import (
    BoolType,
    DictType,
    FloatType,
    IntType,
    ListType,
    LiteralType,
    NodeType,
    NoneType,
    RefType,
    SetType,
    StrType,
    TupleType,
    TypeDef,
    TypeParameter,
    UnionType,
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
            msg = (
                f"Type alias {origin.__name__} expects {len(type_params)} "
                f"arguments but got {len(args)}"
            )
            raise ValueError(msg)
        substitutions = dict(zip(type_params, args, strict=True))
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
            msg = "list type must have an element type"
            raise ValueError(msg)
        return ListType(element=extract_type(args[0]))

    if origin is dict:
        if len(args) != 2:
            msg = "dict type must have key and value types"
            raise ValueError(msg)
        return DictType(key=extract_type(args[0]), value=extract_type(args[1]))

    if origin is set:
        if not args:
            msg = "set type must have an element type"
            raise ValueError(msg)
        return SetType(element=extract_type(args[0]))

    if origin is tuple:
        if not args:
            msg = "tuple type must have element types"
            raise ValueError(msg)
        return TupleType(elements=tuple(extract_type(arg) for arg in args))

    if origin is Literal:
        if not args:
            msg = "Literal type must have values"
            raise ValueError(msg)
        for val in args:
            if not isinstance(val, str | int | bool):
                msg = f"Literal values must be str, int, or bool, got {type(val)}"
                raise TypeError(msg)
        return LiteralType(values=args)

    if origin is not None and isinstance(origin, type) and issubclass(origin, Node):
        return NodeType(extract_type(args[0]) if args else NoneType())

    if isinstance(py_type, type) and issubclass(py_type, Node):
        return NodeType(_extract_node_returns(py_type))

    if origin is Ref:
        return RefType(extract_type(args[0]) if args else NoneType())

    if isinstance(py_type, types.UnionType):
        return UnionType(tuple(extract_type(a) for a in args))

    msg = f"Cannot extract type from: {py_type}"
    raise ValueError(msg)


def _extract_node_returns(cls: type[Node[Any]]) -> TypeDef:
    """Extract the return type from a Node class definition."""
    for base in getattr(cls, "__orig_bases__", ()):
        if origin := get_origin(base):
            if isinstance(origin, type) and issubclass(origin, Node):
                if args := get_args(base):
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
                    ),
                )

    node_fields = (
        FieldSchema(name=f.name, type=extract_type(hints[f.name]))
        for f in fields(cls)
        if not f.name.startswith("_")
    )

    return NodeSchema(
        tag=cls._tag,
        type_params=tuple(type_params),
        returns=_extract_node_returns(cls),
        fields=tuple(node_fields),
    )


def all_schemas() -> dict[str, NodeSchema]:
    """Get all registered node schemas."""
    return {tag: node_schema(cls) for tag, cls in Node.registry.items()}
