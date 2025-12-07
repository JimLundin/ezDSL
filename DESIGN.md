# DSL Node Type System Design

**Target Python Version:** 3.12+ with `from __future__ import annotations`

## Overview

This document describes the design of a type-safe node system for building abstract syntax trees (ASTs) and domain-specific languages (DSLs). The system provides automatic registration, serialization, and schema generation for node types.

The framework enables users to define **nodes** (computation/structure) parameterized by **types** (data containers). All schemas are represented as **dataclasses** which can be serialized to various formats via pluggable adapters.

---

## Core Concepts

### Types vs Nodes

**Types** are data containers. They describe the shape of values that flow between nodes or are embedded within nodes.

```
Types = Python built-ins + User-registered types
      = int, str, float, bool, None, list, dict, ...
      + DataFrame, NDArray, CustomClass, ...
```

**Nodes** are AST elements. They represent computation or structure. Every node is parameterized by the type it produces:

```python
class Add(Node[int]):         # produces int
class Filter[T](Node[list[T]]):  # produces list[T]
class Query(Node[DataFrame]): # produces DataFrame
```

Node fields can be:
- `Node[T]` — a child node producing type T (actual nested node)
- `Ref[Node[T]]` — a reference/pointer to a node producing type T
- `list[Node[T]]` — multiple child nodes
- `T` — an embedded value of type T (data, not computation)

### References: Node[T] vs Ref[Node[T]]

The system supports two ways to connect nodes:

**Direct nesting with `Node[T]`:**
- Embeds the actual child node inline
- Creates a tree structure
- Simple and direct

```python
class Add(Node[float]):
    left: Node[float]   # Actual nested node
    right: Node[float]  # Actual nested node

# Usage
tree = Add(
    left=Literal(5.0),   # Inline node
    right=Literal(3.0)   # Inline node
)
```

**Reference with `Ref[Node[T]]`:**
- Stores a pointer to a node by ID
- Enables graph structures with shared nodes
- Supports cyclic references
- Required when using AST container

```python
class Add(Node[float]):
    left: Ref[Node[float]]   # Reference to a node
    right: Ref[Node[float]]  # Reference to a node

# Usage with AST container
ast = AST(
    root="result",
    nodes={
        "x": Literal(5.0),
        "y": Literal(3.0),
        "sum": Add(left=Ref(id="x"), right=Ref(id="y")),
        "result": Multiply(left=Ref(id="sum"), right=Ref(id="x"))  # Reuses "x"
    }
)

# Resolve references
x_node = ast.resolve(Ref(id="x"))  # Returns: Literal(5.0)
```

**When to use each:**
- Use `Node[T]` for simple trees without sharing
- Use `Ref[Node[T]]` when you need:
  - Shared subexpressions (multiple parents reference same child)
  - Cyclic graphs
  - Explicit graph structure management via AST container

---

## Type System

### Built-in Types

Python built-ins are always available. No registration required:

- **Primitives**: `int`, `float`, `str`, `bool`, `None`
- **Containers**: `list[T]`, `dict[K, V]`, `set[T]`, `tuple[A, B, C]` (fixed-length, heterogeneous)
- **Unions**: `T | U`
- **Literals**: `Literal["a", "b", "c"]` (enumerations)

### External Types (Unregistered)

External types can flow between nodes at runtime without registration:

```python
class DBConnection:
    """External type - no registration needed."""
    pass

class Connect(Node[DBConnection]):
    connection_string: str

class Query(Node[DataFrame]):
    connection: Node[DBConnection]  # just works
```

**External types do NOT need registration** unless you want to:
1. Serialize them as embedded values in nodes
2. Include them in generated schemas/documentation

### Registered Types

Types require registration **only if** they need to be serialized as embedded values in nodes.

**Two registration approaches:**

#### 1. Function Registration (for external types)

Use for types you don't control (e.g., pandas, polars, numpy). Creates an `ExternalType` schema entry:

```python
from typing import overload

@overload
def register[T](
    python_type: type[T],
    *,
    tag: str | None = None,
    encode: Callable[[T], dict],
    decode: Callable[[dict], T],
) -> type[T]: ...

# Example usage
TypeDef.register(
    pd.DataFrame,
    tag="dataframe",
    encode=lambda df: {"data": df.to_dict()},
    decode=lambda d: pd.DataFrame(d["data"]),
)

# Creates ExternalType schema:
# ExternalType(module="pandas.core.frame", name="DataFrame", tag="dataframe")
```

**Key points:**
- Stores module path (e.g., `pandas.core.frame`) and class name to avoid collisions
- `encode` takes instance of type `T`, returns dict
- `decode` takes dict, returns instance of type `T`
- Returns the original type unchanged for chaining

#### 2. Decorator Registration (for custom types)

Use for types you define. Creates a `CustomType` schema entry:

```python
@overload
def register[T](
    python_type: None = None,
    *,
    tag: str | None = None,
) -> Callable[[type[T]], type[T]]: ...

# Example usage
@TypeDef.register(tag="point")
class Point:
    """A 2D point."""
    x: float
    y: float

    def encode(self) -> dict:
        """Convert to dict representation."""
        return {"x": self.x, "y": self.y}

    @classmethod
    def decode(cls, data: dict) -> Self:
        """Construct from dict representation."""
        return cls(x=data["x"], y=data["y"])

# Creates CustomType schema:
# CustomType(tag="point")
```

**Key points:**
- Class must have `encode(self) -> dict` method
- Class must have `decode(cls, data: dict) -> Self` classmethod
- Returns the original class unchanged
- No module/name needed - identified by user-supplied tag

#### 3. Direct TypeDef Subclass (for schema types)

Users can directly subclass TypeDef to create schema types, just like the built-in schema types:

```python
class MySpecialType(TypeDef, tag="myspecial"):
    """User-defined schema type."""
    custom_field: str
    value: int

# Automatically becomes frozen dataclass and registers
# Treated exactly like IntType, FloatType, etc.
```

### Type Registration API

```python
from typing import overload

class TypeDef:
    # Storage for external and custom types
    _external_types: ClassVar[dict[type, ExternalTypeRecord]] = {}
    _custom_types: ClassVar[dict[type, CustomTypeRecord]] = {}

    @overload
    @classmethod
    def register[T](
        cls,
        python_type: type[T],
        *,
        tag: str | None = None,
        encode: Callable[[T], dict],
        decode: Callable[[dict], T],
    ) -> type[T]: ...

    @overload
    @classmethod
    def register[T](
        cls,
        python_type: None = None,
        *,
        tag: str | None = None,
    ) -> Callable[[type[T]], type[T]]: ...

    @classmethod
    def register[T](
        cls,
        python_type: type[T] | None = None,
        *,
        tag: str | None = None,
        encode: Callable[[T], dict] | None = None,
        decode: Callable[[dict], T] | None = None,
    ) -> type[T] | Callable[[type[T]], type[T]]:
        """
        Register a type with the type system.

        Function registration (external types):
            TypeDef.register(
                pd.DataFrame,
                tag="dataframe",
                encode=lambda df: {"data": df.to_dict()},
                decode=lambda d: pd.DataFrame(d["data"])
            )
            Creates ExternalType(module="pandas.core.frame", name="DataFrame", tag="dataframe")

        Decorator registration (custom types):
            @TypeDef.register(tag="point")
            class Point:
                def encode(self) -> dict: ...
                @classmethod
                def decode(cls, data: dict) -> Self: ...
            Creates CustomType(tag="point")
        """
        ...

@dataclass(frozen=True)
class ExternalTypeRecord:
    """Record for external type registration (function style)."""
    python_type: type
    module: str  # Full module path, e.g., "pandas.core.frame"
    name: str    # Class name, e.g., "DataFrame"
    tag: str
    encode: Callable[[Any], dict]
    decode: Callable[[dict], Any]

@dataclass(frozen=True)
class CustomTypeRecord:
    """Record for custom type registration (decorator style)."""
    python_type: type
    tag: str
    # encode/decode are methods on the class, not stored here
```

---

## Node System

### Core Pattern

Nodes use automatic registration via `__init_subclass__`:

- Inherit from `Node[T]`
- Optionally specify `tag` in class definition
- Automatically becomes a frozen dataclass
- Automatically registered in a central registry

### Node Base Class

```python
@dataclass_transform(frozen_default=True)
class Node[T]:
    _tag: ClassVar[str]
    _registry: ClassVar[dict[str, type[Node]]] = {}

    def __init_subclass__(cls, tag: str | None = None):
        dataclass(frozen=True)(cls)

        # Determine tag
        cls._tag = tag or cls.__name__.lower().removesuffix("node")

        # Register by tag
        if existing := Node._registry.get(cls._tag):
            if existing is not cls:
                raise TagCollisionError(
                    f"Tag '{cls._tag}' already registered to {existing}. "
                    f"Choose a different tag."
                )

        Node._registry[cls._tag] = cls
```

**Key Features:**
- Generic type parameter `T` represents the node's return/value type
- `_tag` uniquely identifies the node type for serialization
- `_registry` maps tags to node classes for deserialization
- `__init_subclass__` hook automates dataclass conversion and registration
- `@dataclass_transform` (PEP 681) enables IDE/type checker support
- Frozen by default ensures immutability

### Node Definition

**Python 3.12+ syntax only** (using PEP 695 type parameters):

```python
# Simple node
class Literal(Node[float], tag="literal"):
    value: float

# Generic node (unbounded)
class Map[E, R](Node[list[R]], tag="map"):
    input: Node[list[E]]
    func: Node[R]

# Generic node with bounds
class Add[T: int | float](Node[T], tag="add"):
    left: Node[T]
    right: Node[T]

# Node with multiple type parameters
class MapReduce[E, M, R](Node[R], tag="mapreduce"):
    input: Node[list[E]]
    mapper: Node[M]
    reducer: Node[R]
```

**Not supported**: Legacy `TypeVar` syntax like `T = TypeVar('T', bound=int)`.

### Type Aliases

```python
type NodeRef[T] = Ref[Node[T]]
type Child[T] = Node[T] | Ref[Node[T]]
```

---

## TypeDef Schema Types

TypeDef dataclasses represent type schemas. These are the canonical representation - dict/JSON forms are derived from them.

### TypeDef Base Class

Like Node, TypeDef uses `__init_subclass__` to automatically convert subclasses to dataclasses:

```python
@dataclass_transform(frozen_default=True)
class TypeDef:
    _tag: ClassVar[str]
    _registry: ClassVar[dict[str, type[TypeDef]]] = {}

    def __init_subclass__(cls, tag: str | None = None):
        dataclass(frozen=True)(cls)

        cls._tag = tag or cls.__name__.lower().removesuffix("type")

        if existing := TypeDef._registry.get(cls._tag):
            if existing is not cls:
                raise TagCollisionError(
                    f"Tag '{cls._tag}' already registered to {existing}."
                )

        TypeDef._registry[cls._tag] = cls
```

**No need for `@dataclass` decorator on subclasses!** Just like Node, subclassing automatically makes it a frozen dataclass.

### Primitive Types

```python
class IntType(TypeDef, tag="int"):
    """Integer type."""
    pass

class FloatType(TypeDef, tag="float"):
    """Floating point type."""
    pass

class StrType(TypeDef, tag="str"):
    """String type."""
    pass

class BoolType(TypeDef, tag="bool"):
    """Boolean type."""
    pass

class NoneType(TypeDef, tag="none"):
    """None/null type."""
    pass
```

### Container Types

```python
class ListType(TypeDef, tag="list"):
    """Homogeneous list type."""
    element: TypeDef

class DictType(TypeDef, tag="dict"):
    """Dictionary type with key and value types."""
    key: TypeDef
    value: TypeDef

class SetType(TypeDef, tag="set"):
    """Set type with element type."""
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
```

### Literal/Enumeration Type

```python
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
```

### Domain Types

```python
class NodeType(TypeDef, tag="node"):
    """AST Node type with return type."""
    returns: TypeDef

class RefType(TypeDef, tag="ref"):
    """Reference type pointing to another type."""
    target: TypeDef

class UnionType(TypeDef, tag="union"):
    """Union of multiple types."""
    options: tuple[TypeDef, ...]
```

### Type Variables and References

```python
class TypeVar(TypeDef, tag="typevar"):
    """
    Type variable declaration in PEP 695 generic definitions.

    Represents the DECLARATION of a type variable (e.g., in class Foo[T]).

    Examples:
        class Foo[T]: ... → TypeVar(name="T", bound=None)
        class Foo[T: int | float]: ... → TypeVar(name="T", bound=UnionType(...))

    This is the definition site of the type parameter.
    """
    name: str
    bound: TypeDef | None = None

class TypeVarRef(TypeDef, tag="typevarref"):
    """
    Reference to a type variable within a type expression.

    Represents a USE of a type variable (e.g., in field: T).

    Examples:
        In class Foo[T]:
            field: T → TypeVarRef(name="T")
            field: list[T] → ListType(element=TypeVarRef(name="T"))

    This is the use site that refers back to the TypeVar declaration.
    """
    name: str
```

### External and Custom Types

```python
class ExternalType(TypeDef, tag="external"):
    """
    Reference to an externally registered type.

    Used for types registered via function (e.g., pandas DataFrame, polars DataFrame).
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

    Used for types registered via decorator.
    Identified solely by user-supplied tag.

    Example:
        @TypeDef.register(tag="point")
        class Point: ...
        → CustomType(tag="point")
    """
    tag: str
```

---

## Schema Representation

All schemas are **dataclasses**. They are the canonical representation. Serialization to dict/JSON/YAML/etc happens via format adapters.

### Node Schema

```python
class NodeSchema:
    """Complete schema for a node class."""
    tag: str
    type_params: tuple[TypeVar, ...]  # Type variable declarations
    returns: TypeDef
    fields: tuple[FieldSchema, ...]

class FieldSchema:
    """Schema for a node field."""
    name: str
    type: TypeDef
```

### Schema Conversion Functions

```python
def extract_type(py_type: Any) -> TypeDef:
    """
    Convert a Python type hint to a TypeDef dataclass.

    Examples:
        extract_type(int) → IntType()
        extract_type(list[int]) → ListType(element=IntType())
        extract_type(tuple[int, str]) → TupleType(elements=(IntType(), StrType()))
        extract_type(Literal["a", "b"]) → LiteralType(values=("a", "b"))
        extract_type(Node[float]) → NodeType(returns=FloatType())
        extract_type(pd.DataFrame) → ExternalType(module="...", name="DataFrame", tag="...")
    """
    ...

def node_schema[N: Node](cls: type[N]) -> NodeSchema:
    """
    Extract schema from a Node subclass.

    Returns NodeSchema dataclass, NOT dict.
    """
    ...

def all_schemas() -> dict[str, NodeSchema]:
    """Get all registered node schemas as dataclasses."""
    ...
```

### Schema Examples

**TypeDef examples (dataclasses):**

```python
# int
IntType()

# list[int]
ListType(element=IntType())

# set[str]
SetType(element=StrType())

# dict[str, int]
DictType(key=StrType(), value=IntType())

# tuple[int, str, float] (fixed-length, heterogeneous)
TupleType(elements=(IntType(), StrType(), FloatType()))

# int | str
UnionType(options=(IntType(), StrType()))

# Literal["red", "green", "blue"]
LiteralType(values=("red", "green", "blue"))

# Node[float]
NodeType(returns=FloatType())

# Ref[Node[int]]
RefType(target=NodeType(returns=IntType()))

# pd.DataFrame (externally registered)
ExternalType(module="pandas.core.frame", name="DataFrame", tag="pd_dataframe")

# Point (custom registered)
CustomType(tag="point")
```

**NodeSchema example:**

```python
class Add[T: int | float](Node[T], tag="add"):
    left: Node[T]
    right: Node[T]

# Produces NodeSchema dataclass:
NodeSchema(
    tag="add",
    type_params=(
        TypeVar(
            name="T",
            bound=UnionType(options=(IntType(), FloatType()))
        ),
    ),
    returns=TypeVarRef(name="T"),
    fields=(
        FieldSchema(name="left", type=NodeType(returns=TypeVarRef(name="T"))),
        FieldSchema(name="right", type=NodeType(returns=TypeVarRef(name="T"))),
    )
)
```

---

## Serialization

### Format Adapters

Format adapters convert dataclass schemas to/from specific formats. This is the **only** way to serialize - there are no built-in dict/JSON methods on the schemas.

#### Adapter Interface

```python
from abc import ABC, abstractmethod

class FormatAdapter(ABC):
    """Base class for format-specific serialization."""

    @abstractmethod
    def serialize_node[N: Node](self, node: N) -> Any:
        """Serialize a node instance."""
        ...

    @abstractmethod
    def deserialize_node(self, data: Any) -> Node:
        """Deserialize to a node instance."""
        ...

    @abstractmethod
    def serialize_typedef(self, typedef: TypeDef) -> Any:
        """Serialize a TypeDef dataclass."""
        ...

    @abstractmethod
    def serialize_node_schema(self, schema: NodeSchema) -> Any:
        """Serialize a NodeSchema dataclass."""
        ...
```

#### Built-in Adapters

```python
class JSONAdapter(FormatAdapter):
    """JSON serialization adapter."""

    def serialize_node[N: Node](self, node: N) -> dict:
        """Serialize node to dict."""
        return {
            "tag": type(node)._tag,
            **{
                field.name: self._serialize_value(getattr(node, field.name))
                for field in dataclass_fields(node)
            }
        }

    def serialize_typedef(self, typedef: TypeDef) -> dict:
        """Serialize TypeDef to dict."""
        return {
            "tag": type(typedef)._tag,
            **{
                field.name: self._serialize_value(getattr(typedef, field.name))
                for field in dataclass_fields(typedef)
            }
        }

class YAMLAdapter(FormatAdapter):
    """YAML serialization adapter."""
    ...

class BinaryAdapter(FormatAdapter):
    """Binary serialization adapter."""
    ...
```

#### Usage Example

```python
# Create adapter
json_adapter = JSONAdapter()

# Serialize node
node = Add(left=Literal(1.0), right=Literal(2.0))
data = json_adapter.serialize_node(node)
# → {"tag": "add", "left": {"tag": "literal", "value": 1.0}, "right": {"tag": "literal", "value": 2.0}}

# Deserialize node
restored = json_adapter.deserialize_node(data)

# Serialize schema
schema = node_schema(Add)  # Returns NodeSchema dataclass
schema_json = json_adapter.serialize_node_schema(schema)

# Different format
yaml_adapter = YAMLAdapter()
yaml_str = yaml_adapter.serialize_node(node)
```

### Serialization Format

**Node serialization (inline):**
```python
Add(left=Literal(1.0), right=Literal(2.0))
# Becomes (via JSONAdapter):
{
    "tag": "add",
    "left": {"tag": "literal", "value": 1.0},
    "right": {"tag": "literal", "value": 2.0}
}
```

**Reference serialization:**
```python
Ref(id="node-123")
# Becomes:
{"$ref": "node-123"}
```

**External type serialization:**
```python
# Given pd.DataFrame registered as:
# ExternalType(module="pandas.core.frame", name="DataFrame", tag="pd_dataframe")

df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
# Serializes as:
{"type": "pd_dataframe", "value": {"data": {"a": [1, 2], "b": [3, 4]}}}
```

**Custom type serialization:**
```python
# Given:
@TypeDef.register(tag="point")
class Point:
    x: float
    y: float

    def encode(self) -> dict:
        return {"x": self.x, "y": self.y}

    @classmethod
    def decode(cls, data: dict) -> Self:
        return cls(x=data["x"], y=data["y"])

# Then:
point = Point(x=1.0, y=2.0)
# Serializes as:
{"type": "point", "value": {"x": 1.0, "y": 2.0}}
```

---

## AST Container

Manages the complete abstract syntax tree with node storage and reference resolution.

```python
@dataclass
class AST:
    """Flat AST with nodes stored by ID."""
    root: str
    nodes: dict[str, Node]

    def resolve[X](self, ref: Ref[X]) -> X:
        """Resolve a reference to get the actual node."""
        if ref.id not in self.nodes:
            raise NodeNotFoundError(f"Node '{ref.id}' not found in AST")
        return self.nodes[ref.id]

    def serialize[A: FormatAdapter](self, adapter: A) -> Any:
        """Serialize entire AST using given adapter."""
        return {
            "root": self.root,
            "nodes": {
                node_id: adapter.serialize_node(node)
                for node_id, node in self.nodes.items()
            }
        }

    @classmethod
    def deserialize[A: FormatAdapter](cls, data: Any, adapter: A) -> AST:
        """Deserialize AST using given adapter."""
        return cls(
            root=data["root"],
            nodes={
                node_id: adapter.deserialize_node(node_data)
                for node_id, node_data in data["nodes"].items()
            }
        )
```

---

## Examples

### Logic-Based AST Examples

#### Mathematical Expression with Tuples

```python
class Literal(Node[float], tag="literal"):
    value: float

class Point3D(Node[tuple[float, float, float]], tag="point3d"):
    """Fixed-length heterogeneous tuple."""
    x: Node[float]
    y: Node[float]
    z: Node[float]

class Add(Node[float], tag="add"):
    left: Node[float]
    right: Node[float]

# Build: point with computed coordinates
point = Point3D(
    x=Literal(1.0),
    y=Add(left=Literal(2.0), right=Literal(3.0)),
    z=Literal(4.0)
)

# Schema shows fixed-length tuple
schema = node_schema(Point3D)
# returns: TupleType(elements=(FloatType(), FloatType(), FloatType()))
```

#### Conditional Logic with Literal Enumerations

```python
class StringCase(Node[str], tag="string_case"):
    """Pattern match on string literals."""
    value: Node[str]
    case: Literal["upper", "lower", "title"]  # Enumeration

# Schema extraction
schema = node_schema(StringCase)
# field "case" has type: LiteralType(values=("upper", "lower", "title"))
```

### Data/Structural AST Examples

#### Configuration with Custom Types

```python
# Custom type with encode/decode methods
@TypeDef.register(tag="config_value")
class ConfigValue:
    value: Any
    source: str  # "default", "env", "file"

    def encode(self) -> dict:
        return {"value": self.value, "source": self.source}

    @classmethod
    def decode(cls, data: dict) -> Self:
        return cls(value=data["value"], source=data["source"])

class Config(Node[dict], tag="config"):
    name: str
    settings: dict[str, ConfigValue]  # Embedded custom type

# Schema shows CustomType(tag="config_value")
```

#### Data Pipeline with External Types

```python
# External type registered with module/name to avoid collisions
TypeDef.register(
    pd.DataFrame,
    tag="pd_dataframe",
    encode=lambda df: {"data": df.to_dict()},
    decode=lambda d: pd.DataFrame(d["data"])
)

TypeDef.register(
    pl.DataFrame,  # Polars DataFrame
    tag="pl_dataframe",
    encode=lambda df: {"data": df.to_dict()},
    decode=lambda d: pl.DataFrame(d["data"])
)

# Both pandas and polars DataFrames can coexist
class DataSource(Node[pd.DataFrame], tag="pd_source"):
    path: str

class PolarsSource(Node[pl.DataFrame], tag="pl_source"):
    path: str

# Schemas distinguish via module:
# pd.DataFrame → ExternalType(module="pandas.core.frame", name="DataFrame", tag="pd_dataframe")
# pl.DataFrame → ExternalType(module="polars.dataframe.frame", name="DataFrame", tag="pl_dataframe")
```

---

## Design Principles

1. **Immutability**: All nodes and schemas are frozen dataclasses
2. **Type Safety**: Leverage Python 3.12+ generics with proper type parameters
3. **Automatic Dataclass Conversion**: Both Node and TypeDef auto-convert subclasses
4. **Dataclass-First**: Schemas are dataclasses; serialization is secondary
5. **Minimal Registration**: External types don't need registration unless serialized
6. **Simple Tagging**: Just `tag`, no namespace or version complexity
7. **Modern Python**: PEP 695 type parameters only (`class[T]` syntax)
8. **Generic Signatures**: encode/decode properly typed with type parameters
9. **Module-based Collision Avoidance**: External types store module+name
10. **Three Type Categories**: External (function registration), Custom (decorator registration), Direct (TypeDef subclass)

---

## Type Categories Summary

| Category | How Defined | Schema Type | Registration Required? | Can Embed? |
|----------|-------------|-------------|------------------------|------------|
| Built-in | Python | IntType, ListType, etc. | No | Yes |
| External (unregistered) | Third-party | Not in schema | No | No |
| External (registered) | Function registration | ExternalType(module, name, tag) | For embedding | Yes |
| Custom | Decorator registration | CustomType(tag) | For embedding | Yes |
| Direct | TypeDef subclass | Own type | No | Yes |

---

## Implementation Status

### Currently Implemented

- ✅ Node base class with automatic dataclass conversion
- ✅ TypeDef base class with automatic dataclass conversion
- ✅ Tag validation with regex pattern (lowercase start, lowercase/digits/hyphens/underscores only)
- ✅ Namespace support (to be removed)
- ✅ Generic node support with type parameters
- ✅ Type registration via `TypeDef.register()`
- ✅ Schema extraction (returns dicts)
- ✅ AST container with reference resolution
- ✅ JSONAdapter for serialization/deserialization
- ✅ Comprehensive error handling with helpful error messages
- ✅ Comprehensive test suite (215+ tests covering all core modules)

### Needs Implementation

- ⏳ **TypeVar and TypeVarRef**: Rename TypeParameter for clarity
- ⏳ **LiteralType**: Add support for Python `Literal[...]`
- ⏳ **TupleType semantics**: Document fixed-length heterogeneous tuples
- ⏳ **SetType**: Add schema dataclass
- ⏳ **Dataclass schema returns**: Return NodeSchema instead of dict
- ⏳ **Additional format adapters**: Implement YAMLAdapter, TOMLAdapter, etc.
- ⏳ **Remove string type hints**: Audit and remove any stringified hints
- ⏳ **Simplify to tag-only**: Remove namespace from Node/TypeDef

### Future Features

These features are intentionally excluded from the current implementation for simplicity, but may be added in future versions:

- 🔮 **CustomType with decorator registration**: Support for `@typedsl.register` decorator to register custom Python types
- 🔮 **ExternalType with decorator-based registration**: Decorator approach for registering third-party types
- 🔮 **Generic register signatures**: Type-safe `encode[T]` and `decode[T]` functions with decorators
