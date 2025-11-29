"""
Minimal AST node type system.

Requires Python 3.12+
"""

from __future__ import annotations

from dataclasses import dataclass, fields as dc_fields
from typing import dataclass_transform, get_args, get_origin, get_type_hints, Any, Union, ClassVar
import json

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
# Core Types
# =============================================================================

@dataclass(frozen=True)
class Ref[X]:
    """Reference to X by ID."""
    id: str


type NodeRef[T] = Ref[Node[T]]
type Child[T] = Node[T] | Ref[Node[T]]


@dataclass_transform(frozen_default=True)
class Node[T]:
    """Base for AST nodes. T is return type."""

    _tag: ClassVar[str]
    _registry: ClassVar[dict[str, type[Node]]] = {}

    def __init_subclass__(cls, tag: str | None = None, frozen: bool = True, **kwargs):
        super().__init_subclass__(**kwargs)
        if not cls.__dict__.get("__annotations__"):
            return
        dataclass(frozen=frozen, eq=True, repr=True)(cls)
        cls._tag = tag or cls.__name__.lower()
        Node._registry[cls._tag] = cls


# =============================================================================
# Type Alias Registry
# =============================================================================

from typing import Callable, TypeAliasType

# Helper for type parameter substitution
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


# =============================================================================
# Serialization
# =============================================================================

def to_dict(obj: Node | Ref | TypeDef) -> dict:
    """Serialize to dict."""
    if isinstance(obj, Ref):
        return {"tag": "ref", "id": obj.id}

    tag = getattr(type(obj), "_tag", None)
    if tag is None:
        raise ValueError(f"No tag for {type(obj)}")

    result = {"tag": tag}
    for field in dc_fields(obj):
        result[field.name] = _serialize_value(getattr(obj, field.name))
    return result


def _serialize_value(value: Any) -> Any:
    if isinstance(value, (Node, Ref, TypeDef)):
        return to_dict(value)
    if isinstance(value, tuple):
        return [_serialize_value(v) for v in value]
    if isinstance(value, list):
        return [_serialize_value(v) for v in value]
    if isinstance(value, type):
        return "none" if value is type(None) else value.__name__
    return value


def from_dict(data: dict, registry: dict[str, type] = None) -> Node | Ref | TypeDef:
    """Deserialize from dict."""
    tag = data["tag"]

    if tag == "ref":
        return Ref(id=data["id"])

    # Try Node registry first, then TypeDef
    registry = registry or Node._registry
    cls = registry.get(tag) or TypeDef._registry.get(tag)
    if cls is None:
        raise ValueError(f"Unknown tag: {tag}")

    kwargs = {}
    for field in dc_fields(cls):
        raw = data.get(field.name)
        kwargs[field.name] = _deserialize_value(raw, field.type)
    return cls(**kwargs)


def _deserialize_value(value: Any, hint: Any) -> Any:
    if isinstance(value, dict) and "tag" in value:
        return from_dict(value)
    if isinstance(value, list):
        return tuple(_deserialize_value(v, hint) for v in value)
    if hint is type or (get_origin(hint) is type):
        # Reconstruct type from name
        for p in PRIMITIVES:
            if ("none" if p is type(None) else p.__name__) == value:
                return p
        raise ValueError(f"Unknown primitive: {value}")
    return value


def to_json(obj: Node | Ref | TypeDef) -> str:
    return json.dumps(to_dict(obj), indent=2)


def from_json(s: str) -> Node | Ref | TypeDef:
    return from_dict(json.loads(s))


# =============================================================================
# Schema Extraction
# =============================================================================

def extract_type(py_type: Any) -> TypeDef:
    """Convert Python type annotation to TypeDef."""
    from typing import TypeVar
    import types

    origin = get_origin(py_type)
    args = get_args(py_type)

    # Handle TypeVar (old-style and PEP 695 type parameters)
    if isinstance(py_type, TypeVar):
        bounds = getattr(py_type, "__constraints__", None)
        if bounds:
            return TypeVarType(
                name=py_type.__name__,
                bounds=tuple(extract_type(b) for b in bounds)
            )
        bound = getattr(py_type, "__bound__", None)
        if bound is not None:
            return TypeVarType(
                name=py_type.__name__,
                bounds=(extract_type(bound),)
            )
        return TypeVarType(name=py_type.__name__, bounds=None)

    # PEP 695 type aliases - automatically expand them
    if isinstance(origin, TypeAliasType):
        # Get the type parameters and create substitution mapping
        type_params = getattr(origin, "__type_params__", ())
        if len(type_params) != len(args):
            raise ValueError(
                f"Type alias {origin.__name__} expects {len(type_params)} "
                f"arguments but got {len(args)}"
            )

        # Create substitution mapping: {T: int, U: str, ...}
        substitutions = dict(zip(type_params, args))

        # Get the template and substitute type parameters
        template = origin.__value__
        substituted = _substitute_type_params(template, substitutions)

        # Extract the substituted type
        return extract_type(substituted)

    # Primitives
    if py_type in PRIMITIVES:
        return PrimitiveType(py_type)

    # Node types
    if origin is not None and isinstance(origin, type) and issubclass(origin, Node):
        return NodeType(extract_type(args[0]) if args else PrimitiveType(type(None)))

    if isinstance(py_type, type) and issubclass(py_type, Node):
        return NodeType(_extract_node_returns(py_type))

    # Ref types
    if origin is Ref:
        return RefType(extract_type(args[0]) if args else PrimitiveType(type(None)))

    # Union types (typing.Union)
    if origin is Union:
        return UnionType(tuple(extract_type(a) for a in args))

    # UnionType (types.UnionType, created by | operator in Python 3.10+)
    if isinstance(py_type, types.UnionType):
        return UnionType(tuple(extract_type(a) for a in args))

    # Generic types (fallback for other parameterized types)
    if origin is not None and args:
        origin_name = getattr(origin, "__name__", str(origin))
        return GenericType(
            name=f"{origin_name}[{', '.join(str(a) for a in args)}]",
            origin=origin_name,
            args=tuple(extract_type(a) for a in args)
        )

    raise ValueError(f"Cannot extract type from: {py_type}")


def _extract_node_returns(cls: type[Node]) -> TypeDef:
    for base in getattr(cls, "__orig_bases__", ()):
        origin = get_origin(base)
        if origin is not None and isinstance(origin, type) and issubclass(origin, Node):
            args = get_args(base)
            if args:
                return extract_type(args[0])
    return PrimitiveType(type(None))


def node_schema(cls: type[Node]) -> dict:
    """Get schema for a node class."""
    hints = get_type_hints(cls)
    return {
        "tag": cls._tag,
        "returns": to_dict(_extract_node_returns(cls)),
        "fields": [
            {"name": f.name, "type": to_dict(extract_type(hints[f.name]))}
            for f in dc_fields(cls)
            if not f.name.startswith("_")
        ],
    }


def all_schemas() -> dict:
    """Get all registered node schemas."""
    return {"nodes": {tag: node_schema(cls) for tag, cls in Node._registry.items()}}


# =============================================================================
# AST Container
# =============================================================================

@dataclass
class AST:
    """Flat AST with nodes stored by ID."""
    root: str
    nodes: dict[str, Node]

    def resolve[X](self, ref: Ref[X]) -> X:
        return self.nodes[ref.id]

    def to_dict(self) -> dict:
        return {"root": self.root, "nodes": {k: to_dict(v) for k, v in self.nodes.items()}}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> AST:
        return cls(data["root"], {k: from_dict(v) for k, v in data["nodes"].items()})

    @classmethod
    def from_json(cls, s: str) -> AST:
        return cls.from_dict(json.loads(s))
