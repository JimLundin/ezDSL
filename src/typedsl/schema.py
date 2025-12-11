"""Schema extraction and type reflection utilities."""

from __future__ import annotations

import datetime
import types
from collections.abc import Mapping, Sequence, Set as AbstractSet
from dataclasses import dataclass, fields
from decimal import Decimal
from typing import (
    Any,
    Literal,
    TypeAliasType,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from typedsl.nodes import Node, Ref
from typedsl.types import (
    AbstractSetType,
    BoolType,
    BytesType,
    DateTimeType,
    DateType,
    DecimalType,
    DictType,
    DurationType,
    FloatType,
    FrozenSetType,
    IntType,
    ListType,
    LiteralType,
    MappingType,
    NodeType,
    NoneType,
    RefType,
    SequenceType,
    SetType,
    StrType,
    TimeType,
    TupleType,
    TypeDef,
    TypeParameter,
    UnionType,
    _substitute_type_params,
)

# Unified type map: Python type/origin -> (TypeDef class, expected arg count)
# 0 args = simple type, 1 arg = element container, 2 args = key-value container
_TYPE_MAP: dict[type, tuple[type[TypeDef], int]] = {
    # Simple types (0 args)
    int: (IntType, 0),
    float: (FloatType, 0),
    str: (StrType, 0),
    bool: (BoolType, 0),
    type(None): (NoneType, 0),
    bytes: (BytesType, 0),
    Decimal: (DecimalType, 0),
    datetime.date: (DateType, 0),
    datetime.time: (TimeType, 0),
    datetime.datetime: (DateTimeType, 0),
    datetime.timedelta: (DurationType, 0),
    # Element containers (1 arg)
    list: (ListType, 1),
    set: (SetType, 1),
    frozenset: (FrozenSetType, 1),
    Sequence: (SequenceType, 1),
    AbstractSet: (AbstractSetType, 1),
    # Key-value containers (2 args)
    dict: (DictType, 2),
    Mapping: (MappingType, 2),
}


def _get_type_name(origin: type) -> str:
    """Get a readable name for a type origin."""
    if origin is AbstractSet:
        return "Set"
    return getattr(origin, "__name__", str(origin))


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

    # Handle TypeVar
    if isinstance(py_type, TypeVar):
        bound = py_type.__bound__
        return TypeParameter(
            name=py_type.__name__,
            bound=extract_type(bound) if bound is not None else None,
        )

    # Handle registered external types
    custom_typedef = TypeDef.get_registered_type(py_type)
    if custom_typedef is not None:
        return custom_typedef

    # Expand PEP 695 type aliases
    if isinstance(origin, TypeAliasType):
        type_params = origin.__type_params__
        if len(type_params) != len(args):
            msg = (
                f"Type alias {origin.__name__} expects {len(type_params)} "
                f"arguments but got {len(args)}"
            )
            raise ValueError(msg)
        substitutions = dict(zip(type_params, args, strict=True))
        substituted = _substitute_type_params(origin.__value__, substitutions)
        return extract_type(substituted)

    # Check unified type map (simple types use py_type, containers use origin)
    lookup_key = py_type if py_type in _TYPE_MAP else origin
    if lookup_key in _TYPE_MAP:
        typedef_cls, arg_count = _TYPE_MAP[lookup_key]
        if arg_count == 0:
            return typedef_cls()
        type_name = _get_type_name(lookup_key)
        if len(args) != arg_count:
            if arg_count == 1:
                msg = f"{type_name} type must have an element type"
            else:
                msg = f"{type_name} type must have key and value types"
            raise ValueError(msg)
        if arg_count == 1:
            return typedef_cls(element=extract_type(args[0]))
        return typedef_cls(key=extract_type(args[0]), value=extract_type(args[1]))

    # Tuple (heterogeneous, variable-length elements)
    if origin is tuple:
        if not args:
            msg = "tuple type must have element types"
            raise ValueError(msg)
        return TupleType(elements=tuple(extract_type(arg) for arg in args))

    # Literal values
    if origin is Literal:
        if not args:
            msg = "Literal type must have values"
            raise ValueError(msg)
        for val in args:
            if not isinstance(val, str | int | bool):
                msg = f"Literal values must be str, int, or bool, got {type(val)}"
                raise TypeError(msg)
        return LiteralType(values=args)

    # Node types
    if origin is not None and isinstance(origin, type) and issubclass(origin, Node):
        return NodeType(extract_type(args[0]) if args else NoneType())
    if isinstance(py_type, type) and issubclass(py_type, Node):
        return NodeType(_extract_node_returns(py_type))

    # Ref type
    if origin is Ref:
        return RefType(extract_type(args[0]) if args else NoneType())

    # Union types
    if isinstance(py_type, types.UnionType) or origin is Union:
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
