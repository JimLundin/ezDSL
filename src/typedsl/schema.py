"""Schema extraction and type reflection utilities."""

from __future__ import annotations

import datetime
import types
from collections.abc import Mapping, Sequence
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
    substitute_type_params,
)

# Unified type map: Python type/origin -> TypeDef class
_TYPE_MAP: dict[type, type[TypeDef]] = {
    # Simple types
    int: IntType,
    float: FloatType,
    str: StrType,
    bool: BoolType,
    type(None): NoneType,
    bytes: BytesType,
    Decimal: DecimalType,
    datetime.date: DateType,
    datetime.time: TimeType,
    datetime.datetime: DateTimeType,
    datetime.timedelta: DurationType,
    # Element containers
    list: ListType,
    set: SetType,
    frozenset: FrozenSetType,
    Sequence: SequenceType,
    # Key-value containers
    dict: DictType,
    Mapping: MappingType,
}


def _get_field_count(typedef_cls: type[TypeDef]) -> int:
    """Get the number of public fields from a TypeDef class."""
    return sum(1 for f in fields(typedef_cls) if not f.name.startswith("_"))


@dataclass(frozen=True)
class FieldSchema:
    """Schema for a node field."""

    name: str
    type: TypeDef


@dataclass(frozen=True)
class NodeSchema:
    """Complete schema for a node class."""

    tag: str
    signature: dict[str, Any]  # Signature kwargs (preserves insertion order)
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
        substituted = substitute_type_params(origin.__value__, substitutions)
        return extract_type(substituted)

    # Check unified type map (simple types use py_type, containers use origin)
    lookup_key = py_type if py_type in _TYPE_MAP else origin
    if lookup_key in _TYPE_MAP:
        typedef_cls = _TYPE_MAP[lookup_key]
        field_count = _get_field_count(typedef_cls)
        if len(args) != field_count:
            type_name = getattr(lookup_key, "__name__", str(lookup_key))
            msg = (
                f"{type_name} requires {field_count} type argument(s), got {len(args)}"
            )
            raise ValueError(msg)
        return typedef_cls(*(extract_type(arg) for arg in args))

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
        origin = get_origin(base)
        is_node_origin = isinstance(origin, type) and issubclass(origin, Node)
        if origin is None or not is_node_origin:
            continue
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
        tag=cls.tag,
        signature=cls.signature,
        type_params=tuple(type_params),
        returns=_extract_node_returns(cls),
        fields=tuple(node_fields),
    )


def all_schemas() -> dict[str, NodeSchema]:
    """Get all registered node schemas."""
    return {tag: node_schema(cls) for tag, cls in Node.registry.items()}
