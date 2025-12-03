# DSL Node Type System Design

**Target Python Version:** 3.12+

## Overview

This document describes the design of a type-safe node system for building abstract syntax trees (ASTs) and domain-specific languages (DSLs). The system provides automatic registration, serialization, and schema generation for node types.

The framework enables users to define **nodes** (computation/structure) parameterized by **types** (data containers). Serialization is handled by pluggable **format adapters**.

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
- **Unions**: `T | U` or `Union[T, U]`

### Unregistered Types

Types that only flow between nodes at runtime. **No registration required.** Just use them:

```python
class DBConnection:
    pass

class Connect(Node[DBConnection]):
    connection_string: str

class Query(Node[DataFrame]):
    connection: Node[DBConnection]  # just works
```

Unregistered types can appear in `Node[T]` but **cannot be embedded as values** in node fields.

### Registered Types

Types that need to be embedded as values in nodes require registration with encoding/decoding:

```python
TypeDef.register(
    pd.DataFrame,
    tag="dataframe",
    namespace="custom",
    version="1.0",
    encode=lambda df: {"data": df.to_dict()},
    decode=lambda d: pd.DataFrame(d["data"]),
)
```

After registration, the type can be used both in `Node[T]` and as embedded field values.

### Type Registration API

```python
class TypeDef:
    @classmethod
    def register(
        cls,
        python_type: type | None = None,
        *,
        tag: str | None = None,
        namespace: str = "custom",
        version: str = "1.0",
        encode: Callable[[Any], dict] | None = None,
        decode: Callable[[dict], Any] | None = None,
    ) -> type[TypeDef] | Any:
        """
        Register a custom type with the type system.

        Args:
            python_type: The Python class to register
            tag: Identifier for schema/serialization (defaults to lowercase class name)
            namespace: Namespace for the tag (defaults to "custom")
            version: Version string for the type (defaults to "1.0")
            encode: Function to serialize instances to dict/basic types
            decode: Function to deserialize from dict/basic types

        The full qualified identifier is: {namespace}.{tag}.{version}

        Can be used as decorator or function call:
            @TypeDef.register
            class MyType: ...

            TypeDef.register(
                pd.DataFrame,
                tag="df",
                encode=lambda df: {"data": df.to_dict()},
                decode=lambda d: pd.DataFrame(d["data"])
            )
        """
        ...

    @classmethod
    def get_registered_type(cls, python_type: type) -> type[TypeDef] | None:
        """Get the registered TypeDef for a Python type."""
        ...
```

**Current Implementation Note**: The current implementation supports registration via `TypeDef.register()` but `encode`, `decode`, and `version` parameters are not yet implemented. Registration creates a TypeDef subclass that enables schema generation and serialization of the type reference.

### TypeDef as Schema Representation

TypeDef serves a dual purpose:
1. **Runtime registration** of custom types
2. **Schema representation** for serialization and documentation

TypeDef instances ARE the schema. There is no separate schema type system. The TypeDef hierarchy (IntType, FloatType, ListType, etc.) represents type schemas as frozen dataclasses.

```python
# These TypeDef classes are BOTH runtime types AND schema representations:

class IntType(TypeDef, tag="int", namespace="std"):
    """Integer type - used for both runtime and schema."""
    pass

class ListType(TypeDef, tag="list", namespace="std"):
    """List type with element type."""
    element: TypeDef  # Nested TypeDef = nested schema

class NodeType(TypeDef, tag="node", namespace="std"):
    """Node type with return type."""
    returns: TypeDef
```

When you call `extract_type(int)`, it returns `IntType()` - a TypeDef instance that represents the schema.

### TypeRegistry

The TypeRegistry is a unified view of all registered types, stored on the TypeDef class:

```python
class TypeDef:
    _registry: ClassVar[dict[str, type[TypeDef]]] = {}
    _custom_types: ClassVar[dict[type, type[TypeDef]]] = {}

# _registry maps: qualified_tag -> TypeDef class
# "std.int.1.0" -> IntType
# "custom.dataframe.1.0" -> DataFrameType

# _custom_types maps: Python type -> TypeDef class
# pd.DataFrame -> DataFrameType
# np.ndarray -> NDArrayType
```

This registry enables:
- Looking up TypeDef by qualified tag during deserialization
- Looking up TypeDef by Python type during schema extraction
- Listing all registered types for documentation

---

## Node System

### Core Pattern

Both `Node` and `TypeDef` use the same pattern:

- Inherit from base class
- Optionally specify `tag`, `namespace`, and `version` in class definition
- Automatically becomes a frozen dataclass
- Automatically registered in a central registry

### Node Base Class

```python
@dataclass_transform(frozen_default=True)
class Node[T]:
    _tag: ClassVar[str]              # Base tag (e.g., "add")
    _namespace: ClassVar[str]         # Namespace (e.g., "math")
    _version: ClassVar[str]           # Version (e.g., "1.0")
    _qualified_tag: ClassVar[str]     # Full identifier (e.g., "math.add.1.0")
    _registry: ClassVar[dict[str, type[Node]]] = {}

    def __init_subclass__(
        cls,
        tag: str | None = None,
        namespace: str | None = None,
        version: str = "1.0"
    ):
        dataclass(frozen=True)(cls)

        # Store components
        cls._namespace = namespace or ""
        cls._tag = tag or cls.__name__.lower()
        cls._version = version

        # Create qualified tag: namespace.tag.version
        parts = [p for p in [namespace, cls._tag, version] if p]
        cls._qualified_tag = ".".join(parts)

        # Register by qualified tag
        if existing := Node._registry.get(cls._qualified_tag):
            if existing is not cls:
                raise TagCollisionError(
                    f"Tag '{cls._qualified_tag}' already registered to {existing}. "
                    f"Choose a different tag, namespace, or version."
                )

        Node._registry[cls._qualified_tag] = cls
```

**Key Features:**
- Generic type parameter `T` represents the node's return/value type
- `_tag` is the base identifier (e.g., "add")
- `_namespace` provides organizational grouping
- `_version` enables schema evolution
- `_qualified_tag` is the unique identifier: `{namespace}.{tag}.{version}`
- `_registry` maps qualified tags to node classes for deserialization
- `__init_subclass__` hook automates dataclass conversion and registration
- `@dataclass_transform` (PEP 681) tells type checkers that subclasses will be dataclasses
- Frozen by default ensures immutability

### Namespaces and Versioning

Namespaces and versions are **central features** of the tagging system. Together they create unique qualified tags that prevent collisions and enable schema evolution:

```python
# Different namespaces, same tag and version
class Add(Node[float], tag="add", namespace="math", version="1.0"):
    left: Node[float]
    right: Node[float]
# Qualified tag: "math.add.1.0"

class Add(Node[str], tag="add", namespace="string", version="1.0"):
    parts: list[Node[str]]
# Qualified tag: "string.add.1.0"

# Same namespace and tag, different versions (schema evolution)
class Query(Node[DataFrame], tag="query", namespace="db", version="1.0"):
    sql: str
# Qualified tag: "db.query.1.0"

class Query(Node[DataFrame], tag="query", namespace="db", version="2.0"):
    sql: str
    timeout: float  # New field in v2
# Qualified tag: "db.query.2.0"
```

**Standard Namespaces:**
- `std` — Standard/built-in types (IntType, FloatType, ListType, etc.)
- `custom` — Default namespace for registered user types
- User-defined — Any custom namespace for domain-specific nodes

**Versioning Strategy:**
- Default version: `"1.0"`
- Use semantic versioning (e.g., "1.0", "1.1", "2.0")
- Increment version when making breaking schema changes
- Keep old versions registered for backward compatibility

### Node Definition

```python
# Simple node
class Literal(Node[float], tag="literal", version="1.0"):
    value: float

# Node with child references
class Add(Node[float], tag="add", namespace="math", version="1.0"):
    left: Node[float] | Ref[Node[float]]  # Can be inline or reference
    right: Node[float] | Ref[Node[float]]

# Generic node (unbounded)
class Map[E, R](Node[list[R]], namespace="functional", version="1.0"):
    input: Node[list[E]]
    func: Node[R]

# Generic node with bounds (Python 3.12+ syntax)
class Add[T: int | float](Node[T], namespace="math", version="2.0"):
    left: Node[T]
    right: Node[T]

# Generic node with bounds (older syntax)
T = TypeVar('T', bound=int | float)

class Add(Node[T], namespace="math", version="1.0"):
    left: Node[T]
    right: Node[T]
```

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
| `T` (registered type) | Embedded value | Depends on type's encode function |

---

## TypeDef Schema Types

TypeDef includes built-in schema types for representing all Python type annotations:

### Primitive Types

```python
class IntType(TypeDef, tag="int", namespace="std", version="1.0"):
    """Integer type."""
    pass

class FloatType(TypeDef, tag="float", namespace="std", version="1.0"):
    """Floating point type."""
    pass

class StrType(TypeDef, tag="str", namespace="std", version="1.0"):
    """String type."""
    pass

class BoolType(TypeDef, tag="bool", namespace="std", version="1.0"):
    """Boolean type."""
    pass

class NoneType(TypeDef, tag="none", namespace="std", version="1.0"):
    """None/null type."""
    pass
```

### Container Types

```python
class ListType(TypeDef, tag="list", namespace="std", version="1.0"):
    """List type with element type."""
    element: TypeDef

class DictType(TypeDef, tag="dict", namespace="std", version="1.0"):
    """Dictionary type with key and value types."""
    key: TypeDef
    value: TypeDef

class SetType(TypeDef, tag="set", namespace="std", version="1.0"):
    """Set type with element type."""
    element: TypeDef

class TupleType(TypeDef, tag="tuple", namespace="std", version="1.0"):
    """Tuple type with fixed element types."""
    elements: tuple[TypeDef, ...]  # Fixed-length tuple
```

### Domain Types

```python
class NodeType(TypeDef, tag="node", namespace="std", version="1.0"):
    """AST Node type with return type."""
    returns: TypeDef

class RefType(TypeDef, tag="ref", namespace="std", version="1.0"):
    """Reference type pointing to another type."""
    target: TypeDef

class UnionType(TypeDef, tag="union", namespace="std", version="1.0"):
    """Union of multiple types."""
    options: tuple[TypeDef, ...]

class TypeParameter(TypeDef, tag="param", namespace="std", version="1.0"):
    """Type parameter in generic definitions."""
    name: str
    bound: TypeDef | None = None  # Upper bound constraint
```

### Custom Type Definitions

When you register a custom type, a TypeDef subclass is created:

```python
# User registers DataFrame
TypeDef.register(pd.DataFrame, tag="dataframe", namespace="custom", version="1.0")

# System creates:
class DataFrameType(TypeDef, tag="dataframe", namespace="custom", version="1.0"):
    """Custom type definition for DataFrame."""
    pass

# Usage in schemas
df_type = DataFrameType()  # Instance represents the schema
```

---

## Schema Extraction and Node Schema

### Schema Conversion Functions

```python
def extract_type(py_type: Any) -> TypeDef:
    """
    Convert a Python type hint to a TypeDef instance.

    Examples:
        extract_type(int) -> IntType()
        extract_type(list[int]) -> ListType(element=IntType())
        extract_type(Node[float]) -> NodeType(returns=FloatType())
    """
    ...

def node_schema(cls: type[Node]) -> dict:
    """
    Get schema for a node class.

    Returns dict with structure:
    {
        "tag": str,           # Base tag
        "namespace": str,     # Namespace
        "version": str,       # Version
        "qualified_tag": str, # Full identifier
        "returns": dict,      # Serialized TypeDef
        "fields": [
            {"name": str, "type": dict},  # Serialized TypeDef
            ...
        ]
    }
    """
    ...

def all_schemas() -> dict:
    """Get all registered node schemas."""
    ...
```

### Schema Examples

**Simple type schemas:**
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

# Node[float]
NodeType(returns=FloatType())

# Ref[Node[int]]
RefType(target=NodeType(returns=IntType()))
```

**Node schema example:**

```python
class Add[T: int | float](Node[T], namespace="math", version="1.0"):
    left: Node[T]
    right: Node[T]

# Produces schema:
{
    "tag": "add",
    "namespace": "math",
    "version": "1.0",
    "qualified_tag": "math.add.1.0",
    "type_params": [
        {
            "name": "T",
            "bound": {
                "tag": "union",
                "options": [
                    {"tag": "int"},
                    {"tag": "float"}
                ]
            }
        }
    ],
    "returns": {"tag": "param", "name": "T"},
    "fields": [
        {
            "name": "left",
            "type": {
                "tag": "node",
                "returns": {"tag": "param", "name": "T"}
            }
        },
        {
            "name": "right",
            "type": {
                "tag": "node",
                "returns": {"tag": "param", "name": "T"}
            }
        }
    ]
}
```

---

## Serialization

### Current Implementation

Simple, consistent serialization API. Pattern is `{"tag": qualified_tag, **fields}`:

#### API

```python
to_dict(obj)   # Node | Ref | TypeDef -> dict
to_json(obj)   # Node | Ref | TypeDef -> str
from_dict(d)   # dict -> Node | Ref | TypeDef
from_json(s)   # str -> Node | Ref | TypeDef
```

#### Serialization Format

**Node serialization (inline):**
```python
Add(left=Literal(1.0), right=Literal(2.0))
# Becomes:
{
    "tag": "math.add.1.0",
    "left": {"tag": "literal.1.0", "value": 1.0},
    "right": {"tag": "literal.1.0", "value": 2.0}
}
```

**Reference serialization:**
```python
Ref(id="node-123")
# Becomes:
{"$ref": "node-123"}
```

**TypeDef serialization:**
```python
ListType(element=IntType())
# Becomes:
{"tag": "std.list.1.0", "element": {"tag": "std.int.1.0"}}
```

#### Deserialization

Uses qualified tag to lookup node/type class from registry:
```python
data = {"tag": "math.add.1.0", "left": {...}, "right": {...}}
node = from_dict(data)
# Looks up Node._registry["math.add.1.0"] -> Add class
# Recursively deserializes fields
# Returns: Add(left=..., right=...)
```

### Format Adapters (Future Design)

Adapters will handle serialization to/from specific formats as a pluggable system. This replaces the current JSON-only focus.

#### Adapter Interface

```python
from abc import ABC, abstractmethod

class FormatAdapter(ABC):
    @abstractmethod
    def serialize_node(
        self,
        node: Node,
        type_registry: TypeRegistry,
    ) -> Any:
        """Serialize a node instance to the output format."""
        ...

    @abstractmethod
    def deserialize_node(
        self,
        data: Any,
        node_registry: dict[str, type[Node]],
        type_registry: TypeRegistry,
    ) -> Node:
        """Deserialize from the input format to a node instance."""
        ...

    @abstractmethod
    def serialize_type(self, type_def: TypeDef) -> Any:
        """Serialize a TypeDef to the output format."""
        ...

    @abstractmethod
    def serialize_schema(self, schema: dict) -> Any:
        """Serialize a node schema to the output format."""
        ...
```

#### Usage Example

```python
# JSON adapter
json_adapter = JSONAdapter()
json_data = json_adapter.serialize_node(my_node, TypeDef._registry)

# YAML adapter
yaml_adapter = YAMLAdapter()
yaml_str = yaml_adapter.serialize_node(my_node, TypeDef._registry)

# Binary adapter
binary_adapter = BinaryAdapter()
binary_data = binary_adapter.serialize_node(my_node, TypeDef._registry)
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

    def to_dict(self) -> dict:
        """Serialize entire AST to dict."""
        return {
            "root": self.root,
            "nodes": {k: to_dict(v) for k, v in self.nodes.items()},
        }

    def to_json(self) -> str:
        """Serialize entire AST to JSON."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> AST:
        """Deserialize AST from dict."""
        return cls(
            root=data["root"],
            nodes={k: from_dict(v) for k, v in data["nodes"].items()}
        )

    @classmethod
    def from_json(cls, s: str) -> AST:
        """Deserialize AST from JSON."""
        return cls.from_dict(json.loads(s))
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
Occurs when two node/type classes try to register with the same qualified tag:

```python
class Add(Node[int], tag="add", namespace="math", version="1.0"):
    pass

class Add(Node[float], tag="add", namespace="math", version="1.0"):
    pass
# Raises: TagCollisionError("Tag 'math.add.1.0' already registered...")
```

**Solution**: Use different tag, namespace, or version.

#### Invalid Type Annotation
Occurs when a type annotation cannot be processed:

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
Occurs when trying to serialize a node with an unregistered type as embedded value:

```python
class UnregisteredClass:
    pass

class BadNode(Node[int]):
    value: UnregisteredClass  # Not registered!

node = BadNode(value=UnregisteredClass())
to_dict(node)  # Raises: UnregisteredTypeError(
               #   "Cannot embed unregistered type UnregisteredClass. "
               #   "Use TypeDef.register() to register this type."
               # )
```

**Solution**: Register the type with `TypeDef.register()` or use it only in `Node[T]` position (not as embedded value).

#### Unknown Tag During Deserialization
Occurs when deserializing data with a tag not in the registry:

```python
data = {"tag": "unknown.node.1.0", "value": 42}
from_dict(data)  # Raises: UnknownTagError("Tag 'unknown.node.1.0' not found in registry")
```

**Solution**: Ensure all node classes are imported/registered before deserialization.

### Reference Errors

#### Node Not Found
Occurs when resolving a reference to a non-existent node:

```python
ast = AST(root="main", nodes={"main": ...})
ast.resolve(Ref(id="missing"))  # Raises: NodeNotFoundError("Node 'missing' not found in AST")
```

**Solution**: Ensure all referenced nodes are present in the AST container.

### Validation Errors (Future)

Future validation hooks will enable custom validation:

```python
class PositiveNumber(Node[float]):
    value: float

    def __validate__(self):
        if self.value <= 0:
            raise ValidationError("Value must be positive")
```

---

## Examples

### Logic-Based AST Examples

#### 1. Mathematical Expression Tree

```python
# Define nodes
class Literal(Node[float], tag="literal", namespace="math"):
    value: float

class Add(Node[float], tag="add", namespace="math"):
    left: Node[float]
    right: Node[float]

class Multiply(Node[float], tag="multiply", namespace="math"):
    left: Node[float]
    right: Node[float]

class Negate(Node[float], tag="negate", namespace="math"):
    operand: Node[float]

# Build expression: -(2 + 3) * 4
expr = Multiply(
    left=Negate(
        operand=Add(
            left=Literal(2.0),
            right=Literal(3.0)
        )
    ),
    right=Literal(4.0)
)

# Serialize
data = to_json(expr)
# Deserialize
restored = from_json(data)
```

#### 2. Conditional Logic

```python
class If[T](Node[T], tag="if", namespace="logic"):
    condition: Node[bool]
    then_branch: Node[T]
    else_branch: Node[T]

class Equal[T](Node[bool], tag="equal", namespace="logic"):
    left: Node[T]
    right: Node[T]

class GreaterThan(Node[bool], tag="gt", namespace="logic"):
    left: Node[float]
    right: Node[float]

# if x > 0 then x * 2 else -x
x = Literal(5.0)
conditional = If(
    condition=GreaterThan(left=x, right=Literal(0.0)),
    then_branch=Multiply(left=x, right=Literal(2.0)),
    else_branch=Negate(operand=x)
)
```

#### 3. Generic Map/Filter/Reduce

```python
class Map[E, R](Node[list[R]], tag="map", namespace="functional"):
    input: Node[list[E]]
    func: Node[R]

class Filter[T](Node[list[T]], tag="filter", namespace="functional"):
    input: Node[list[T]]
    predicate: Node[bool]

class Reduce[E, A](Node[A], tag="reduce", namespace="functional"):
    input: Node[list[E]]
    accumulator: A
    func: Node[A]

# Example usage
numbers = ListLiteral([1, 2, 3, 4, 5])
doubled = Map(
    input=numbers,
    func=Lambda(body=Multiply(left=Var("x"), right=Literal(2)))
)
```

### Data/Structural AST Examples

#### 1. Document Structure (HTML-like)

```python
# Register custom types for attributes
@TypeDef.register
class Attributes:
    """HTML attributes."""
    pass

class Element(Node['Element'], tag="element", namespace="html"):
    tag_name: str
    attributes: dict[str, str]
    children: list[Node['Element'] | TextNode]

class TextNode(Node[str], tag="text", namespace="html"):
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

#### 2. Configuration Tree

```python
class Config(Node[dict], tag="config", namespace="app"):
    name: str
    version: str
    settings: dict[str, Node[Any]]

class DatabaseConfig(Node[dict], tag="database", namespace="app"):
    host: str
    port: int
    credentials: Node['Credentials']

class Credentials(Node[dict], tag="credentials", namespace="app"):
    username: str
    password_ref: str  # Reference to secret store

# Build config
config = Config(
    name="MyApp",
    version="1.0",
    settings={
        "database": DatabaseConfig(
            host="localhost",
            port=5432,
            credentials=Credentials(
                username="admin",
                password_ref="secret://db/password"
            )
        )
    }
)
```

#### 3. Schema Definition (Like JSON Schema)

```python
class SchemaNode(Node[dict], tag="schema", namespace="schema"):
    title: str
    description: str
    type_def: Node['TypeSpec']

class ObjectSpec(Node[dict], tag="object", namespace="schema"):
    properties: dict[str, Node['TypeSpec']]
    required: list[str]

class StringSpec(Node[dict], tag="string", namespace="schema"):
    min_length: int | None = None
    max_length: int | None = None
    pattern: str | None = None

class NumberSpec(Node[dict], tag="number", namespace="schema"):
    minimum: float | None = None
    maximum: float | None = None

# Define schema for a User object
user_schema = SchemaNode(
    title="User",
    description="User account information",
    type_def=ObjectSpec(
        properties={
            "username": StringSpec(min_length=3, max_length=20),
            "age": NumberSpec(minimum=0, maximum=150),
            "email": StringSpec(pattern=r"^[^@]+@[^@]+\.[^@]+$")
        },
        required=["username", "email"]
    )
)
```

#### 4. Build System DAG

```python
class Task(Node[str], tag="task", namespace="build"):
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
        "test": Task(
            name="test",
            command="./run_tests.sh",
            dependencies=[Ref(id="link")],
            outputs=["test_results.xml"]
        ),
        "deploy": Task(
            name="deploy",
            command="./deploy.sh",
            dependencies=[Ref(id="test"), Ref(id="link")],
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

#### 5. Data Pipeline Graph

```python
@TypeDef.register
class DataFrame:
    """Pandas DataFrame."""
    pass

class DataSource(Node[DataFrame], tag="source", namespace="data"):
    path: str
    format: str  # "csv", "parquet", etc.

class Transform(Node[DataFrame], tag="transform", namespace="data"):
    input: Ref[Node[DataFrame]]
    operation: str
    params: dict[str, Any]

class Join(Node[DataFrame], tag="join", namespace="data"):
    left: Ref[Node[DataFrame]]
    right: Ref[Node[DataFrame]]
    on: str
    how: str  # "inner", "left", "outer"

class DataSink(Node[None], tag="sink", namespace="data"):
    input: Ref[Node[DataFrame]]
    path: str
    format: str

# Build pipeline
pipeline = AST(
    root="output",
    nodes={
        "users": DataSource(path="users.csv", format="csv"),
        "orders": DataSource(path="orders.csv", format="csv"),
        "filtered_orders": Transform(
            input=Ref(id="orders"),
            operation="filter",
            params={"condition": "amount > 100"}
        ),
        "joined": Join(
            left=Ref(id="users"),
            right=Ref(id="filtered_orders"),
            on="user_id",
            how="inner"
        ),
        "aggregated": Transform(
            input=Ref(id="joined"),
            operation="groupby",
            params={"by": ["user_id"], "agg": {"amount": "sum"}}
        ),
        "output": DataSink(
            input=Ref(id="aggregated"),
            path="user_totals.parquet",
            format="parquet"
        )
    }
)
```

---

## Design Principles

1. **Immutability**: All nodes are frozen dataclasses
2. **Type Safety**: Leverage Python 3.12+ generics for compile-time type checking
3. **Automatic Registration**: No manual registry management
4. **Uniform Pattern**: Same approach for Node and TypeDef
5. **Namespace-based Organization**: Prevent collisions and provide structure
6. **Versioning Support**: Enable schema evolution and backward compatibility
7. **Minimal Ceremony**: Unregistered types work without registration
8. **Unified Schema Representation**: TypeDef instances serve as both runtime types and schemas
9. **Pluggable Serialization**: Format adapters separate concerns
10. **Reference Support**: First-class support for node references and graph structures

---

## Type Categories Summary

| Category | Registration Required? | Can be Node[T] parameter? | Can be embedded value? | Example |
|----------|------------------------|---------------------------|------------------------|---------|
| Built-in types | No | Yes | Yes | `int`, `list[str]` |
| Unregistered types | No | Yes | No | `DBConnection` |
| Registered types | Yes | Yes | Yes | `DataFrame` |

---

## Implementation Status

### Currently Implemented

- ✅ Node base class with automatic registration
- ✅ TypeDef base class with automatic registration
- ✅ Namespace support for both nodes and types
- ✅ Generic node support with type parameters
- ✅ Type registration via `TypeDef.register()`
- ✅ Schema extraction to dicts via `node_schema()` and `all_schemas()`
- ✅ Serialization via `to_dict/from_dict/to_json/from_json`
- ✅ AST container with reference resolution
- ✅ Unregistered type support (any type can be used in Node[T])
- ✅ TypeDef as unified schema representation

### Future Work

- ⏳ **Version support**: Add `version` parameter to Node and TypeDef registration
- ⏳ **Encoding/decoding functions**: Implement `encode` and `decode` parameters in `TypeDef.register()`
- ⏳ **SetType and TupleType**: Add schema types for `set[T]` and `tuple[T, ...]`
- ⏳ **Format adapter interface**: Implement pluggable serialization system
- ⏳ **Validation hooks**: Add `__validate__()` method for custom validation
- ⏳ **Error types**: Define specific exception classes (TagCollisionError, UnregisteredTypeError, etc.)
- ⏳ **Node traversal utilities**: Add helpers for walking/transforming ASTs
- ⏳ **Type inference system**: Infer concrete types for generic nodes
- ⏳ **Pretty printing and visualization**: Tools for displaying ASTs

---

## Migration Strategy

### Phase 1: Current State
- `node_schema()` returns dicts
- TypeDef instances used internally
- JSON-only serialization

### Phase 2: Add Versioning
- Add `version` parameter to `__init_subclass__`
- Update registry to use qualified tags with version
- Maintain backward compatibility with version="1.0" default

### Phase 3: Implement Encoding/Decoding
- Add `encode` and `decode` to `TypeDef.register()`
- Support serialization of embedded custom type values
- Add tests for round-trip serialization

### Phase 4: Format Adapters
- Implement `FormatAdapter` interface
- Create JSONAdapter wrapping current implementation
- Add YAMLAdapter, BinaryAdapter, etc.
- Provide migration guide for users

### Phase 5: Enhanced Schemas
- Add SetType and TupleType
- Add validation hooks
- Implement schema versioning and migration tools
