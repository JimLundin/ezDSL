# DSL Node Type System Design

**Target Python Version:** 3.12+

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
class Filter(Node[list[T]]):  # produces list[T]
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
- **Containers**: `list[T]`, `dict[K, V]`, `set[T]`, `tuple[T, ...]`
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

#### 1. Function Registration (for existing classes)

Use for types you don't control (e.g., pandas, numpy):

```python
# Encode: takes instance, returns dict
# Decode: takes dict, returns instance
TypeDef.register(
    pd.DataFrame,
    tag="dataframe",
    encode=lambda df: {"data": df.to_dict()},
    decode=lambda d: pd.DataFrame(d["data"]),
)
```

#### 2. Decorator Registration (for your own classes)

Use for types you define:

```python
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
```

The decorated class must have:
- `encode(self) -> dict` method
- `decode(cls, data: dict) -> Self` classmethod

### Type Registration API

```python
class TypeDef:
    # Storage for registered types
    _registry: ClassVar[dict[str, 'RegisteredType']] = {}

    @classmethod
    def register(
        cls,
        python_type: type | None = None,
        *,
        tag: str | None = None,
        encode: Callable[[Any], dict] | None = None,
        decode: Callable[[dict], Any] | None = None,
    ) -> type | Any:
        """
        Register a custom type with the type system.

        Args:
            python_type: The Python class to register
            tag: Identifier for serialization (defaults to lowercase class name)
            encode: Function to serialize instance to dict
            decode: Function to deserialize from dict to instance

        For function registration:
            - encode takes instance, returns dict
            - decode takes dict, returns instance

        For decorator registration:
            - Class must have encode(self) -> dict method
            - Class must have decode(cls, data: dict) -> Self classmethod

        Examples:
            # Function registration (external types)
            TypeDef.register(
                pd.DataFrame,
                tag="dataframe",
                encode=lambda df: {"data": df.to_dict()},
                decode=lambda d: pd.DataFrame(d["data"])
            )

            # Decorator registration (your types)
            @TypeDef.register(tag="point")
            class Point:
                x: float
                y: float

                def encode(self) -> dict:
                    return {"x": self.x, "y": self.y}

                @classmethod
                def decode(cls, data: dict) -> Self:
                    return cls(x=data["x"], y=data["y"])
        """
        ...

    @classmethod
    def get_registered_type(cls, python_type: type) -> 'RegisteredType' | None:
        """Get registration info for a Python type."""
        ...
```

**Implementation Note**: Instead of dynamically creating TypeDef subclasses, the system stores a `RegisteredType` record:

```python
@dataclass(frozen=True)
class RegisteredType:
    """Record of a registered custom type."""
    python_type: type
    tag: str
    encode: Callable[[Any], dict]
    decode: Callable[[dict], Any]
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

**Purpose:**
- `NodeRef[T]`: Explicitly represents a reference to a node
- `Child[T]`: Convenient union type for inline nodes or references

### Field Types

| Annotation | Meaning | Serialization |
|------------|---------|---------------|
| `Node[T]` | Actual nested child node | Inline: full node serialized |
| `Ref[Node[T]]` | Reference to a node | Reference: `{"$ref": "node-id"}` |
| `Node[T] \| Ref[Node[T]]` | Either inline or reference | Depends on value |
| `list[Node[T]]` | Multiple children | List of inline nodes |
| `T` (registered type) | Embedded value | Uses type's encode function |

---

## TypeDef Schema Types

TypeDef dataclasses represent type schemas. These are the canonical representation - dict/JSON forms are derived from them.

### Primitive Types

```python
@dataclass(frozen=True)
class IntType(TypeDef, tag="int"):
    """Integer type."""
    pass

@dataclass(frozen=True)
class FloatType(TypeDef, tag="float"):
    """Floating point type."""
    pass

@dataclass(frozen=True)
class StrType(TypeDef, tag="str"):
    """String type."""
    pass

@dataclass(frozen=True)
class BoolType(TypeDef, tag="bool"):
    """Boolean type."""
    pass

@dataclass(frozen=True)
class NoneType(TypeDef, tag="none"):
    """None/null type."""
    pass
```

### Container Types

```python
@dataclass(frozen=True)
class ListType(TypeDef, tag="list"):
    """List type with element type."""
    element: TypeDef

@dataclass(frozen=True)
class DictType(TypeDef, tag="dict"):
    """Dictionary type with key and value types."""
    key: TypeDef
    value: TypeDef

@dataclass(frozen=True)
class SetType(TypeDef, tag="set"):
    """Set type with element type."""
    element: TypeDef

@dataclass(frozen=True)
class TupleType(TypeDef, tag="tuple"):
    """Tuple type with fixed element types."""
    elements: tuple[TypeDef, ...]
```

### Literal/Enumeration Type

```python
@dataclass(frozen=True)
class LiteralType(TypeDef, tag="literal"):
    """
    Literal type representing enumeration of values.

    Maps Python's Literal[...] type to enumeration schema.
    Example: Literal["red", "green", "blue"] → LiteralType(values=("red", "green", "blue"))

    Note: Does not support Python enum.Enum at this stage.
    """
    values: tuple[str | int | bool, ...]
```

### Domain Types

```python
@dataclass(frozen=True)
class NodeType(TypeDef, tag="node"):
    """AST Node type with return type."""
    returns: TypeDef

@dataclass(frozen=True)
class RefType(TypeDef, tag="ref"):
    """Reference type pointing to another type."""
    target: TypeDef

@dataclass(frozen=True)
class UnionType(TypeDef, tag="union"):
    """Union of multiple types."""
    options: tuple[TypeDef, ...]

@dataclass(frozen=True)
class TypeParameter(TypeDef, tag="param"):
    """
    Type parameter in PEP 695 generic definitions.

    Example: class Foo[T: int | float] → TypeParameter(name="T", bound=UnionType(...))
    """
    name: str
    bound: TypeDef | None = None
```

### Registered Custom Types

```python
@dataclass(frozen=True)
class CustomType(TypeDef, tag="custom"):
    """Reference to a user-registered type."""
    type_tag: str  # The tag used when registering
    python_type: type  # The actual Python type
```

---

## Schema Representation

All schemas are **dataclasses**. They are the canonical representation. Serialization to dict/JSON/YAML/etc happens via format adapters.

### Node Schema

```python
@dataclass(frozen=True)
class NodeSchema:
    """Complete schema for a node class."""
    tag: str
    type_params: tuple[TypeVarDef, ...]  # Type parameters from class[T, U, ...]
    returns: TypeDef
    fields: tuple[FieldSchema, ...]

@dataclass(frozen=True)
class FieldSchema:
    """Schema for a node field."""
    name: str
    type: TypeDef

@dataclass(frozen=True)
class TypeVarDef:
    """
    Type variable definition from PEP 695 syntax.

    Example:
        class Foo[T]: ... → TypeVarDef(name="T", bound=None)
        class Foo[T: int | float]: ... → TypeVarDef(name="T", bound=UnionType(...))
    """
    name: str
    bound: TypeDef | None = None
```

### Schema Conversion Functions

```python
def extract_type(py_type: Any) -> TypeDef:
    """
    Convert a Python type hint to a TypeDef dataclass.

    Examples:
        extract_type(int) → IntType()
        extract_type(list[int]) → ListType(element=IntType())
        extract_type(Literal["a", "b"]) → LiteralType(values=("a", "b"))
        extract_type(Node[float]) → NodeType(returns=FloatType())
    """
    ...

def node_schema(cls: type[Node]) -> NodeSchema:
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

# tuple[int, str, float]
TupleType(elements=(IntType(), StrType(), FloatType()))

# int | str
UnionType(options=(IntType(), StrType()))

# Literal["red", "green", "blue"]
LiteralType(values=("red", "green", "blue"))

# Node[float]
NodeType(returns=FloatType())

# Ref[Node[int]]
RefType(target=NodeType(returns=IntType()))
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
        TypeVarDef(
            name="T",
            bound=UnionType(options=(IntType(), FloatType()))
        ),
    ),
    returns=TypeParameter(name="T"),
    fields=(
        FieldSchema(name="left", type=NodeType(returns=TypeParameter(name="T"))),
        FieldSchema(name="right", type=NodeType(returns=TypeParameter(name="T"))),
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
    def serialize_node(self, node: Node) -> Any:
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

    def serialize_node(self, node: Node) -> dict:
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

**Registered type serialization:**
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

    def serialize(self, adapter: FormatAdapter) -> Any:
        """Serialize entire AST using given adapter."""
        return {
            "root": self.root,
            "nodes": {
                node_id: adapter.serialize_node(node)
                for node_id, node in self.nodes.items()
            }
        }

    @classmethod
    def deserialize(cls, data: Any, adapter: FormatAdapter) -> 'AST':
        """Deserialize AST using given adapter."""
        return cls(
            root=data["root"],
            nodes={
                node_id: adapter.deserialize_node(node_data)
                for node_id, node_data in data["nodes"].items()
            }
        )
```

**Responsibilities:**
- Store all nodes in a flat dictionary keyed by ID
- Provide reference resolution via `resolve()`
- Maintain a single root entry point
- Enable serialization of cyclic graphs and shared subexpressions

**When to use AST container:**
- Complex graphs with shared nodes
- Cyclic references
- Need to serialize/deserialize entire graph at once
- Want explicit graph structure management

**When to use inline nodes:**
- Simple tree structures
- No sharing or cycles
- Direct, simple construction

---

## Error Handling

### Registration Errors

#### Tag Collision
```python
class Add(Node[int], tag="add"):
    pass

class Add(Node[float], tag="add"):
    pass
# Raises: TagCollisionError("Tag 'add' already registered...")
```

**Solution**: Use different tag.

#### Invalid Type Annotation
```python
class BadNode(Node[int]):
    value: SomeUnknownType  # Cannot extract schema
# Raises: InvalidTypeError("Cannot extract type from: SomeUnknownType")
```

**Solution**: Ensure all field types are either:
- Built-in types
- Registered custom types
- Valid Node/Ref types

### Serialization Errors

#### Unregistered Embedded Type
```python
class UnregisteredClass:
    pass

class BadNode(Node[int]):
    value: UnregisteredClass  # Not registered!

node = BadNode(value=UnregisteredClass())
adapter.serialize_node(node)  # Raises: UnregisteredTypeError
```

**Solution**: Register the type with `TypeDef.register()` or use it only in `Node[T]` position.

#### Unknown Tag During Deserialization
```python
data = {"tag": "unknown", "value": 42}
adapter.deserialize_node(data)  # Raises: UnknownTagError
```

**Solution**: Ensure all node classes are imported/registered before deserialization.

### Reference Errors

#### Node Not Found
```python
ast = AST(root="main", nodes={"main": ...})
ast.resolve(Ref(id="missing"))  # Raises: NodeNotFoundError
```

**Solution**: Ensure all referenced nodes are present in the AST container.

---

## Examples

### Logic-Based AST Examples

#### 1. Mathematical Expression Tree

```python
# Define nodes
class Literal(Node[float], tag="literal"):
    value: float

class Add(Node[float], tag="add"):
    left: Node[float]
    right: Node[float]

class Multiply(Node[float], tag="multiply"):
    left: Node[float]
    right: Node[float]

# Build expression: (2 + 3) * 4
expr = Multiply(
    left=Add(left=Literal(2.0), right=Literal(3.0)),
    right=Literal(4.0)
)

# Serialize
adapter = JSONAdapter()
data = adapter.serialize_node(expr)
# Deserialize
restored = adapter.deserialize_node(data)
```

#### 2. Conditional Logic with Literal Types

```python
class If[T](Node[T], tag="if"):
    condition: Node[bool]
    then_branch: Node[T]
    else_branch: Node[T]

class StringCase(Node[str], tag="string_case"):
    """Pattern match on string literals."""
    value: Node[str]
    # Use Literal type for allowed cases
    case: Literal["upper", "lower", "title"]

# Build: if condition then uppercase else lowercase
conditional = If(
    condition=some_bool_node,
    then_branch=StringCase(value=text, case="upper"),
    else_branch=StringCase(value=text, case="lower")
)
```

#### 3. Generic Map/Filter

```python
class Map[E, R](Node[list[R]], tag="map"):
    input: Node[list[E]]
    func: Node[R]

class Filter[T](Node[list[T]], tag="filter"):
    input: Node[list[T]]
    predicate: Node[bool]

# Example: filter then map
numbers = ListLiteral([1, 2, 3, 4, 5])
evens = Filter(input=numbers, predicate=IsEven())
doubled = Map(input=evens, func=Double())
```

### Data/Structural AST Examples

#### 1. Document Structure (HTML-like)

```python
class Element(Node['Element'], tag="element"):
    tag_name: str
    attributes: dict[str, str]
    children: list[Node['Element'] | Node[str]]

class TextNode(Node[str], tag="text"):
    content: str

# Build document: <div class="container"><p>Hello</p></div>
doc = Element(
    tag_name="div",
    attributes={"class": "container"},
    children=[
        Element(
            tag_name="p",
            attributes={},
            children=[TextNode(content="Hello")]
        )
    ]
)
```

#### 2. Configuration with Registered Types

```python
# Register config value type
@TypeDef.register(tag="config_value")
class ConfigValue:
    """Configuration value with metadata."""
    value: Any
    source: str  # "default", "env", "file"

    def encode(self) -> dict:
        return {"value": self.value, "source": self.source}

    @classmethod
    def decode(cls, data: dict) -> Self:
        return cls(value=data["value"], source=data["source"])

class Config(Node[dict], tag="config"):
    name: str
    version: str
    settings: dict[str, ConfigValue]  # Embedded registered type

# Build config
config = Config(
    name="MyApp",
    version="1.0",
    settings={
        "timeout": ConfigValue(value=30, source="default"),
        "api_key": ConfigValue(value="secret", source="env")
    }
)
```

#### 3. Build System DAG

```python
class Task(Node[str], tag="task"):
    name: str
    command: str
    dependencies: list[Ref[Node[str]]]  # References to other tasks
    outputs: list[str]

# Build DAG with AST container
build_graph = AST(
    root="deploy",
    nodes={
        "compile": Task(
            name="compile",
            command="gcc -c main.c",
            dependencies=[],
            outputs=["main.o"]
        ),
        "link": Task(
            name="link",
            command="gcc -o app main.o",
            dependencies=[Ref(id="compile")],
            outputs=["app"]
        ),
        "deploy": Task(
            name="deploy",
            command="./deploy.sh",
            dependencies=[Ref(id="link")],
            outputs=[]
        )
    }
)

# Traverse dependencies
deploy_task = build_graph.resolve(Ref(id="deploy"))
for dep_ref in deploy_task.dependencies:
    dep_task = build_graph.resolve(dep_ref)
    print(f"Deploy depends on: {dep_task.name}")
```

#### 4. Data Pipeline with External Types

```python
# External type - no registration needed since not embedded as value
class DataFrame:
    """External DataFrame type."""
    pass

class DataSource(Node[DataFrame], tag="source"):
    path: str
    format: Literal["csv", "parquet", "json"]  # Use Literal for enums

class Transform(Node[DataFrame], tag="transform"):
    input: Ref[Node[DataFrame]]
    operation: str
    params: dict[str, Any]

# Build pipeline
pipeline = AST(
    root="output",
    nodes={
        "users": DataSource(path="users.csv", format="csv"),
        "filtered": Transform(
            input=Ref(id="users"),
            operation="filter",
            params={"condition": "age > 18"}
        ),
        "output": Transform(
            input=Ref(id="filtered"),
            operation="select",
            params={"columns": ["name", "age"]}
        )
    }
)
```

---

## Design Principles

1. **Immutability**: All nodes and schemas are frozen dataclasses
2. **Type Safety**: Leverage Python 3.12+ generics for compile-time type checking
3. **Automatic Registration**: No manual registry management
4. **Dataclass-First**: Schemas are dataclasses; serialization is secondary
5. **Minimal Registration**: External types don't need registration unless serialized
6. **Simple Tagging**: Just `tag`, no namespace or version complexity
7. **Modern Python**: PEP 695 type parameters only (`class[T]` syntax)
8. **Pluggable Serialization**: Format adapters handle all serialization
9. **Two Registration Styles**: Function for external types, decorator for owned types
10. **Reference Support**: First-class support for node references and graph structures

---

## Type Categories Summary

| Category | Registration Required? | Can be Node[T] parameter? | Can be embedded value? | Example |
|----------|------------------------|---------------------------|------------------------|---------|
| Built-in types | No | Yes | Yes | `int`, `list[str]`, `Literal["a", "b"]` |
| External types | No (unless embedded) | Yes | Only if registered | `DBConnection`, `DataFrame` |
| Registered types | Yes (for embedding) | Yes | Yes | Registered `Point`, `ConfigValue` |

---

## Implementation Status

### Currently Implemented

- ✅ Node base class with automatic registration
- ✅ TypeDef base class with automatic registration
- ✅ Namespace support (to be simplified to tag-only)
- ✅ Generic node support with type parameters
- ✅ Type registration via `TypeDef.register()`
- ✅ Schema extraction via `node_schema()` and `all_schemas()`
- ✅ Serialization via `to_dict/from_dict/to_json/from_json` (to be replaced with adapters)
- ✅ AST container with reference resolution

### Needs Implementation

- ⏳ **Simplify to tag-only**: Remove namespace and version from Node/TypeDef
- ⏳ **RegisteredType record**: Replace dynamic TypeDef generation with simple record
- ⏳ **Encode/decode in registration**: Support both function and method styles
- ⏳ **LiteralType**: Add support for Python `Literal[...]` type
- ⏳ **SetType and TupleType**: Add schema dataclasses for these containers
- ⏳ **Dataclass schemas**: Return NodeSchema dataclasses instead of dicts
- ⏳ **Format adapters**: Implement JSONAdapter, YAMLAdapter, etc.
- ⏳ **Remove legacy TypeVar**: Only support `class[T]` syntax
- ⏳ **Error types**: Define specific exception classes
- ⏳ **Node traversal utilities**: Add helpers for walking/transforming ASTs
- ⏳ **Pretty printing and visualization**: Tools for displaying ASTs
