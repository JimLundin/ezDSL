# typeDSL Tutorial

Learn typeDSL by building knowledge progressively, feature by feature.

**Time**: ~30 minutes
**Prerequisites**: Python 3.12+, basic understanding of classes and type hints

---

## What You'll Learn

- Define typed AST nodes
- Use type parameters for polymorphic nodes
- Build interpreters with pattern matching
- Work with references and shared nodes
- Serialize and deserialize ASTs

---

## Part 1: Your First Node (5 min)

### Concept: Nodes are Building Blocks

Nodes are the fundamental units of your AST. Each node is parameterized by the type it produces.

```python
from typedsl import Node

class NumberLiteral(Node[int]):
    """A literal integer value."""
    value: int

# Create a node
num = NumberLiteral(value=42)
print(num)  # NumberLiteral(value=42)
```

### What Happened Automatically

When you subclass `Node[T]`:
- ✓ Converted to a frozen dataclass (immutable)
- ✓ Registered in the central registry with tag `"numberliteral"`
- ✓ Ready for serialization

### The Type Parameter Matters

```python
class NumberLiteral(Node[int]):  # ← This node "produces" an int
    value: int

class StringLiteral(Node[str]):  # ← This node "produces" a string
    value: str
```

The type parameter `[T]` tells you what type of value this node represents or evaluates to.

### Exercise

Create `StringLiteral` and `BoolLiteral` node classes:

```python
class StringLiteral(Node[str]):
    value: str

class BoolLiteral(Node[bool]):
    value: bool
```

---

## Part 2: Composing Nodes (5 min)

### Concept: Nodes Contain Other Nodes

Build tree structures by composing nodes together.

```python
class Add(Node[int]):
    """Addition of two integers."""
    left: Node[int]
    right: Node[int]

# Build an AST: 5 + 3
expr = Add(
    left=NumberLiteral(value=5),
    right=NumberLiteral(value=3)
)
```

### Understanding the Structure

This creates a tree:
```
    Add
   /   \
  5     3
```

- **Type safety**: `left` and `right` must be `Node[int]`
- **Immutability**: Once created, cannot be modified
- **Composable**: Trees can be arbitrarily deep

### Exercise

Add more operations:

```python
class Multiply(Node[int]):
    left: Node[int]
    right: Node[int]

class Subtract(Node[int]):
    left: Node[int]
    right: Node[int]

# Build: (10 - 3) * 2
expr = Multiply(
    left=Subtract(
        left=NumberLiteral(value=10),
        right=NumberLiteral(value=3)
    ),
    right=NumberLiteral(value=2)
)
```

---

## Part 2.5: Generic Nodes (8 min)

### Concept: Write Once, Use with Any Type

Type parameters make nodes **polymorphic** - they work with different types.

```python
class Container[T](Node[list[T]]):
    """A container that holds items of type T."""
    items: list[T]

# Use with different types
int_container = Container[int](items=[1, 2, 3])
str_container = Container[str](items=["a", "b", "c"])
```

### Why Type Parameters?

**Without generics** (specific to int):
```python
class IntAdd(Node[int]):
    left: Node[int]
    right: Node[int]

class FloatAdd(Node[float]):
    left: Node[float]
    right: Node[float]

# Need separate classes for each type!
```

**With generics** (works for multiple types):
```python
class Add[T: int | float](Node[T]):
    """Addition that works for int OR float."""
    left: Node[T]
    right: Node[T]

# One class, multiple types
int_add = Add[int](left=int_node, right=int_node)
float_add = Add[float](left=float_node, right=float_node)
```

### Type Constraints

Constrain which types are allowed:

```python
class Add[T: int | float](Node[T]):
    #      ^^^^^^^^^^^^^ T must be int or float
    left: Node[T]
    right: Node[T]

# This works
Add[int](...)   # ✓
Add[float](...) # ✓

# This fails type checking
Add[str](...)   # ✗ Error: str not in (int | float)
```

### Real-World Example: Generic Operations

```python
from typing import Literal

class BinOp[T: int | float](Node[T]):
    """Binary operation on numbers."""
    op: Literal["+", "-", "*", "/"]
    left: Node[T]
    right: Node[T]

class Compare[T](Node[bool]):
    """Compare two values of the same type."""
    op: Literal["==", "!=", "<", ">", "<=", ">="]
    left: Node[T]
    right: Node[T]

# Type-safe comparisons
age_check = Compare[int](
    op=">=",
    left=age_node,
    right=NumberLiteral(value=18)
)

# Type error: can't compare int to str
broken = Compare[???](  # What type?
    left=age_node,      # int
    right=name_node     # str - doesn't match!
)
```

### Exercise

Create a generic `Pair[A, B]` node that holds two values of different types:

```python
class Pair[A, B](Node[tuple[A, B]]):
    first: Node[A]
    second: Node[B]

# Use it
pair = Pair[int, str](
    first=NumberLiteral(value=42),
    second=StringLiteral(value="answer")
)
```

---

## Part 3: Evaluating with Interpreters (10 min)

### Concept: Traverse and Evaluate Your AST

Interpreters walk through your AST and compute results.

```python
from typedsl import Interpreter

class Calculator(Interpreter[None, int]):
    """
    Evaluates arithmetic expressions.

    Type parameters:
    - Context: None (no environment needed)
    - Result: int (returns an integer)
    """

    def eval(self, node: Node[int]) -> int:
        """Evaluate a single node using pattern matching."""
        match node:
            case NumberLiteral(value=v):
                return v

            case Add(left=l, right=r):
                return self.eval(l) + self.eval(r)

            case Multiply(left=l, right=r):
                return self.eval(l) * self.eval(r)

            case Subtract(left=l, right=r):
                return self.eval(l) - self.eval(r)

            case _:
                raise NotImplementedError(f"Unknown node: {type(node)}")

# Build expression: (10 - 3) * 2
expr = Multiply(
    left=Subtract(
        left=NumberLiteral(value=10),
        right=NumberLiteral(value=3)
    ),
    right=NumberLiteral(value=2)
)

# Evaluate
calculator = Calculator(None, None)
result = calculator.eval(expr)
print(result)  # 14
```

### Understanding Pattern Matching

The `match` statement dispatches to different code based on node type:

```python
match node:
    case NumberLiteral(value=v):  # If it's a NumberLiteral, extract value
        return v

    case Add(left=l, right=r):    # If it's Add, extract left and right
        return self.eval(l) + self.eval(r)
```

### Interpreter Type Parameters

```python
class Interpreter[Ctx, R]:
    #                ^^^  ^^
    #                |    └─ Result type
    #                └────── Context type
```

- **`Ctx`**: Environment/state passed to interpreter (e.g., variables, config)
- **`R`**: Type returned by `run()` method

### With Context (Variables)

```python
class Variable(Node[int]):
    name: str

class Calculator(Interpreter[dict[str, int], int]):
    #                         ^^^^^^^^^^^^^^  Context is a dict

    def eval(self, node: Node[int]) -> int:
        match node:
            case Variable(name=n):
                if n not in self.ctx:
                    raise ValueError(f"Undefined variable: {n}")
                return self.ctx[n]

            case NumberLiteral(value=v):
                return v

            case Add(left=l, right=r):
                return self.eval(l) + self.eval(r)

# Evaluate with variables
expr = Add(
    left=Variable(name="x"),
    right=NumberLiteral(value=10)
)

calculator = Calculator(None, {"x": 5})
result = calculator.eval(expr)  # Returns 15
```

---

## Part 4: References and Shared Nodes (8 min)

### Concept: Use References for Graph Structures

For DAGs (directed acyclic graphs) and shared nodes, use `Ref`.

#### Inline Nodes (Trees)

```python
class Add(Node[int]):
    left: Node[int]   # Direct child
    right: Node[int]

# Each node is embedded
expr = Add(
    left=NumberLiteral(value=5),
    right=NumberLiteral(value=3)
)
```

#### Reference-Based (Graphs)

```python
from typedsl import Ref, AST

class Add(Node[int]):
    left: Ref[Node[int]]   # Reference by ID
    right: Ref[Node[int]]

# Build: (x + 5) * (x + 5) - x is shared
ast = AST(
    root="result",
    nodes={
        "x": NumberLiteral(value=3),
        "five": NumberLiteral(value=5),
        "sum": Add(left=Ref(id="x"), right=Ref(id="five")),
        "result": Multiply(left=Ref(id="sum"), right=Ref(id="sum"))
    }
)
```

### When to Use References

| Feature | Inline Nodes | References |
|---------|-------------|------------|
| **Structure** | Trees only | Trees or DAGs |
| **Sharing** | No | Yes |
| **Explicit IDs** | No | Yes |
| **Serialization** | Node-by-node | Entire AST |

Use `Ref` when you need:
- **Shared subexpressions**: Multiple nodes reference the same child
- **Explicit node IDs**: For debugging or external references
- **DAG structures**: Not just trees

### Updating the Interpreter

With references, use `resolve()` to get the actual node:

```python
class Calculator(Interpreter[dict[str, int], int]):
    def eval(self, node: Node[int]) -> int:
        match node:
            case NumberLiteral(value=v):
                return v

            case Add(left=l, right=r):
                # Resolve references first
                left_node = self.resolve(l)
                right_node = self.resolve(r)
                return self.eval(left_node) + self.eval(right_node)

# Evaluate
calculator = Calculator(ast, {})
result = calculator.run()  # Starts from root
print(result)  # 64 = (3 + 5) * (3 + 5)
```

### The AST Container

```python
class AST:
    root: str                          # ID of root node
    nodes: dict[str, Node[Any]]        # All nodes by ID

    def resolve[X](self, ref: Ref[X]) -> X:
        """Get node by reference."""
        return self.nodes[ref.id]
```

---

## Part 5: Serialization (4 min)

### Concept: Save and Load ASTs

Serialize ASTs to JSON for storage, transmission, or debugging.

```python
# Serialize to JSON
json_str = ast.to_json()
print(json_str)

# Output:
# {
#   "root": "result",
#   "nodes": {
#     "x": {"tag": "numberliteral", "value": 3},
#     "five": {"tag": "numberliteral", "value": 5},
#     "sum": {"tag": "add", "left": {"$ref": "x"}, "right": {"$ref": "five"}},
#     "result": {"tag": "multiply", "left": {"$ref": "sum"}, "right": {"$ref": "sum"}}
#   }
# }
```

### Deserialization

```python
from typedsl import AST

# Load from JSON
restored_ast = AST.from_json(json_str)

# They're equal
assert restored_ast == ast

# Can evaluate the restored AST
calculator = Calculator(restored_ast, {})
result = calculator.run()
```

### Round-Tripping

Ensure information is preserved:

```python
# Original -> JSON -> Restored
original_ast = build_ast()
json_str = original_ast.to_json()
restored_ast = AST.from_json(json_str)

# Should be equal
assert restored_ast == original_ast

# Should evaluate the same
result1 = Calculator(original_ast, {}).run()
result2 = Calculator(restored_ast, {}).run()
assert result1 == result2
```

---

## Part 6: Putting It All Together (5 min)

### Complete Example: Calculator with Variables

```python
from typedsl import Node, Ref, AST, Interpreter
from typing import Literal

# 1. Define nodes
class Const(Node[int], tag="const"):
    value: int

class Var(Node[int], tag="var"):
    name: str

class BinOp[T: int | float](Node[T], tag="binop"):
    op: Literal["+", "-", "*", "/"]
    left: Ref[Node[T]]
    right: Ref[Node[T]]

# 2. Build AST: (x + 10) * (x + 10)
ast = AST(
    root="result",
    nodes={
        "x": Var(name="x"),
        "ten": Const(value=10),
        "sum": BinOp[int](op="+", left=Ref(id="x"), right=Ref(id="ten")),
        "result": BinOp[int](op="*", left=Ref(id="sum"), right=Ref(id="sum"))
    }
)

# 3. Implement interpreter
class Calculator(Interpreter[dict[str, int], int]):
    def eval(self, node: Node[int]) -> int:
        match node:
            case Const(value=v):
                return v

            case Var(name=n):
                return self.ctx[n]

            case BinOp(op="+", left=l, right=r):
                return self.eval(self.resolve(l)) + self.eval(self.resolve(r))

            case BinOp(op="*", left=l, right=r):
                return self.eval(self.resolve(l)) * self.eval(self.resolve(r))

# 4. Evaluate
calculator = Calculator(ast, {"x": 5})
result = calculator.run()
print(f"Result: {result}")  # 225 = (5 + 10) * (5 + 10)

# 5. Serialize
with open("calculator.json", "w") as f:
    f.write(ast.to_json())

# 6. Load and re-evaluate
with open("calculator.json") as f:
    loaded_ast = AST.from_json(f.read())

calculator2 = Calculator(loaded_ast, {"x": 7})
result2 = calculator2.run()
print(f"Result with x=7: {result2}")  # 289 = (7 + 10) * (7 + 10)
```

---

## Next Steps

Now that you understand the fundamentals:

1. **Explore Examples**: See `examples/` for complete working code
2. **Problem-Based Tutorials**:
   - `EXPRESSION_LANGUAGE.md` - Build a custom expression language
   - `SQL_QUERY_BUILDER.md` - Build a type-safe SQL query builder
3. **Read Design Doc**: See `DESIGN.md` for architectural details
4. **Build Your Own DSL**: Apply these concepts to your domain

---

## Key Takeaways

✓ **Nodes are immutable, type-safe building blocks**
✓ **Type parameters enable polymorphic, reusable nodes**
✓ **Interpreters use pattern matching for evaluation**
✓ **References enable DAG structures and node sharing**
✓ **Serialization preserves entire AST structure**

Happy DSL building! 🚀
