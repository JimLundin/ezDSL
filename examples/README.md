# nanoDSL Examples

This directory contains comprehensive examples demonstrating how to use nanoDSL to build Domain-Specific Languages (DSLs) and work with Abstract Syntax Trees (ASTs).

## Overview

Each example is a standalone, runnable Python script that demonstrates different aspects of nanoDSL:

| Example | Focus | Key Concepts |
|---------|-------|--------------|
| [01_calculator.py](#1-calculator-dsl) | Basic DSL implementation | Node definition, Interpreter class, inline vs. reference-based ASTs, memoization |
| [02_configuration_dsl.py](#2-configuration-dsl) | Validation patterns | Custom validation, error handling, context management |
| [03_workflow_dsl.py](#3-workflow-dsl) | Complex graph structures | DAG construction, diamond patterns, node sharing, caching |
| [04_adapting_existing_ast.py](#4-adapting-existing-asts) | AST adaptation | Converting external ASTs, preserving semantics, round-tripping |

## Prerequisites

- Python 3.12+ (required for PEP 695 generic syntax)
- nanoDSL installed (`pip install nanodsl` or install from source)

## Running the Examples

Each example can be run independently:

```bash
# Run a specific example
python examples/01_calculator.py

# Or use the module syntax
python -m examples.01_calculator
```

All examples include:
- **Detailed docstrings** explaining the concepts
- **Multiple sub-examples** demonstrating different use cases
- **Step-by-step progression** from simple to complex
- **Console output** showing results and explanations

## Example Details

### 1. Calculator DSL

**File**: `01_calculator.py`

**What you'll learn**:
- How to define AST nodes for a DSL
- The difference between inline nodes (`Node[T]`) and references (`Ref[Node[T]]`)
- How to use the `Interpreter` base class for evaluation
- Pattern matching for clean node evaluation
- Building both tree and graph structures
- Implementing memoization for shared subexpressions

**Key sections**:
1. **Node Definition** - Defining Const, Var, BinOp, UnaryOp nodes
2. **Inline Trees** - Building expression trees with direct nesting
3. **Interpreter Implementation** - Evaluating expressions with pattern matching
4. **Reference-Based ASTs** - Using `AST` container and `Ref` for graphs
5. **Memoization** - Caching results for efficient DAG evaluation

**Example expressions**:
- `abs(x + 2) * 3` - Nested operations
- `(x + y) * (x + y)` - Shared subexpressions
- `(sum * 2) + (sum * 3)` - Diamond pattern with memoization

### 2. Configuration DSL

**File**: `02_configuration_dsl.py`

**What you'll learn**:
- Modeling configuration structures as AST nodes
- Implementing custom validation rules
- Using context objects to pass validation state
- Accumulating and reporting validation errors
- Environment variable resolution
- Nested configuration groups

**Key sections**:
1. **Configuration Nodes** - String, integer, boolean values, env vars, groups
2. **Validation Rules** - Required fields, range checks, pattern matching, enums
3. **Validation Context** - Environment variables, rules, error accumulation
4. **Interpreter with Validation** - Evaluating configs while checking rules
5. **Error Handling** - Collecting and reporting multiple validation errors

**Example configurations**:
- **Valid configuration** - Database and server settings with env vars
- **Invalid configuration** - Missing required fields, out-of-range values
- **Custom validation** - Business logic validation (e.g., min < max)

### 3. Workflow DSL

**File**: `03_workflow_dsl.py`

**What you'll learn**:
- Building DAG (Directed Acyclic Graph) structures
- Working with shared nodes (diamond patterns)
- Implementing data pipeline operations
- Caching node results to avoid recomputation
- Dependency resolution in complex graphs
- Serializing and deserializing workflows

**Key sections**:
1. **Workflow Nodes** - DataSource, Transform, Filter, Join, Aggregate, Output
2. **Execution Context** - Mock data sources, transformation functions
3. **Workflow Interpreter** - DAG execution with caching
4. **Pipeline Examples** - Linear, diamond, and complex multi-source workflows

**Example workflows**:
- **Linear pipeline**: source → filter → output
- **Diamond pattern**: Shared source with multiple transformation paths
- **Complex DAG**: Multiple sources, joins, filters, and aggregations
- **Serialization**: Round-tripping workflows through JSON

### 4. Adapting Existing ASTs

**File**: `04_adapting_existing_ast.py`

**What you'll learn**:
- Mapping external AST structures to nanoDSL nodes
- Converting between different AST representations
- Preserving type information and semantics
- Building interpreters for adapted ASTs
- Round-tripping: external → nanoDSL → JSON → nanoDSL → evaluate
- Best practices for AST adaptation

**Key sections**:
1. **Node Mapping** - Creating nanoDSL nodes matching Python's AST
2. **Converter Implementation** - Traversing and converting external ASTs
3. **Interpreter** - Evaluating converted ASTs
4. **Examples** - Simple expressions, variables, statements, complex expressions
5. **Round-Trip Testing** - Verifying information preservation

**Example conversions**:
- `2 + 3 * 4` - Simple arithmetic
- `x + y * 2` - Expressions with variables
- `x = 5; y = x + 3` - Multiple statements
- `(a + b) * (c - d) / 2` - Complex nested expressions

This example uses Python's built-in `ast` module, but the patterns apply to **any existing AST system**:
- Tree-sitter parsers
- ANTLR-generated parsers
- Custom or proprietary AST formats
- Language-specific ASTs (JavaScript, SQL, etc.)

## Common Patterns Across Examples

### Pattern 1: Inline vs. Reference-Based Nodes

**Inline nodes** (direct nesting):
```python
class BinOp(Node[float]):
    left: Node[float]   # Direct child node
    right: Node[float]

# Usage
expr = BinOp(left=Const(1.0), right=Const(2.0))
```

**Reference-based nodes** (for graphs):
```python
class BinOp(Node[float]):
    left: Ref[Node[float]]   # Reference to a node by ID
    right: Ref[Node[float]]

# Usage with AST container
ast = AST(
    root="result",
    nodes={
        "x": Const(value=1.0),
        "y": Const(value=2.0),
        "result": BinOp(left=Ref(id="x"), right=Ref(id="y"))
    }
)
```

**When to use each**:
- Use **inline** for simple tree structures without sharing
- Use **references** when you need:
  - Shared subexpressions (DAGs)
  - Explicit node IDs
  - Serialization to/from JSON

### Pattern 2: Interpreter Implementation

All examples follow this pattern:

```python
class MyInterpreter(Interpreter[ContextType, ReturnType]):
    """
    Type parameters:
    - ContextType: Environment/state passed to interpreter
    - ReturnType: Type returned by run()
    """

    def eval(self, node: Node[Any]) -> Any:
        """Evaluate a single node using pattern matching."""
        match node:
            case NodeType1(field1=f1, field2=f2):
                # Handle NodeType1
                return ...

            case NodeType2(field=f):
                # Resolve references if needed
                child = self.resolve(f)
                # Recursively evaluate
                return self.eval(child)

            case _:
                raise NotImplementedError(f"Unknown node: {type(node)}")

# Usage
interpreter = MyInterpreter(ast, context)
result = interpreter.run()  # Evaluates from root
```

### Pattern 3: Validation Context

For DSLs that need validation:

```python
@dataclass
class ValidationContext:
    """Context with validation state."""
    validation_rules: list[ValidationRule]
    errors: list[str]

    def add_error(self, error: str) -> None:
        self.errors.append(error)

    def has_errors(self) -> bool:
        return len(self.errors) > 0

# In interpreter
class MyInterpreter(Interpreter[ValidationContext, ResultType]):
    def eval(self, node: Node[Any]) -> Any:
        # Evaluate and validate
        if validation_fails:
            self.ctx.add_error("Validation error message")
        return result
```

### Pattern 4: Memoization for DAGs

When nodes are shared (DAGs), cache results:

```python
class MemoizedInterpreter(Interpreter[Context, Result]):
    def __init__(self, ast: AST, ctx: Context) -> None:
        super().__init__(ast, ctx)
        self._cache: dict[str, Any] = {}

    def eval_ref(self, ref: Ref[Node[T]]) -> T:
        """Evaluate reference with caching."""
        if ref.id not in self._cache:
            self._cache[ref.id] = self.eval(self.resolve(ref))
        return self._cache[ref.id]
```

## Design Tips

### 1. Start Simple

Begin with basic node types and add complexity incrementally:
1. Define core nodes (literals, variables, basic operations)
2. Implement a simple interpreter
3. Add more node types as needed
4. Introduce validation, optimization, etc.

### 2. Use Type Parameters Wisely

Each `Node[T]` should be parameterized by what it produces:
```python
class IntLiteral(Node[int]):     # Produces int
class StringExpr(Node[str]):     # Produces str
class Query(Node[DataFrame]):    # Produces DataFrame
class Statement(Node[None]):     # Produces nothing (side effects)
```

### 3. Leverage Pattern Matching

Python 3.10+ pattern matching makes interpreters clean:
```python
match node:
    case BinOp(op="+", left=l, right=r):
        return self.eval(l) + self.eval(r)

    case BinOp(op="-", left=l, right=r):
        return self.eval(l) - self.eval(r)
```

### 4. Handle Errors Gracefully

Provide helpful error messages:
```python
case Var(name=n):
    if n not in self.ctx:
        raise ValueError(
            f"Undefined variable: {n}\n"
            f"Available variables: {list(self.ctx.keys())}"
        )
    return self.ctx[n]
```

### 5. Test Round-Tripping

Verify serialization preserves semantics:
```python
# Original
ast = build_ast()

# Serialize
json_str = ast.to_json()

# Deserialize
restored = AST.from_json(json_str)

# Verify
assert restored == ast
assert evaluate(restored) == evaluate(ast)
```

## Next Steps

After working through these examples, you should be able to:

1. **Define your own DSL** - Model domain-specific concepts as AST nodes
2. **Implement evaluation** - Build interpreters to execute your DSL
3. **Add validation** - Ensure DSL programs are valid before execution
4. **Handle complexity** - Work with DAGs, shared nodes, and complex structures
5. **Integrate existing systems** - Adapt external ASTs to nanoDSL

### Further Reading

- **[Main README](../README.md)** - Package overview and quick start
- **[DESIGN.md](../DESIGN.md)** - Detailed design documentation
- **[API Reference](../README.md#api-reference)** - Core classes and functions
- **[Tests](../tests/)** - Additional usage examples in test suite

## Contributing

Found an issue or have a suggestion for a new example? Please [open an issue](https://github.com/JimLundin/nanoDSL/issues) or submit a pull request!

## License

These examples are licensed under the same terms as nanoDSL: Apache-2.0 / MIT
