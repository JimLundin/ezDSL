# Tutorial: Building a Custom Expression Language

Build a complete typed expression language from scratch with variables, functions, and type checking.

**Time**: 40 minutes
**Difficulty**: Intermediate
**Prerequisites**: Complete `TUTORIAL.md` first

---

## What We're Building

A typed expression language with:
- Variables and literals (int, bool)
- Generic arithmetic operations
- Function definitions and calls
- Let bindings
- Type checking before evaluation

**Example expression:**
```
let double = fn(x) => x * 2 in
let result = double(5) + 3 in
result
```

**Result**: `13`

---

## Step 1: Define Core Nodes (10 min)

### Literals

```python
from typedsl import Node, Ref, AST, Interpreter
from typing import Any, Literal

class IntLit(Node[int], tag="intlit"):
    """Integer literal."""
    value: int

class BoolLit(Node[bool], tag="boollit"):
    """Boolean literal."""
    value: bool
```

### Variables

```python
class Var(Node[Any], tag="var"):
    """Variable reference - type determined at evaluation."""
    name: str
```

### Generic Binary Operations

This is where type parameters shine:

```python
class BinOp[T: int | bool](Node[T], tag="binop"):
    """
    Binary operation on int or bool.

    Type parameter T ensures both operands have the same type.
    """
    op: Literal["+", "-", "*", "/", "==", "!=", "<", ">", "and", "or"]
    left: Ref[Node[T]]
    right: Ref[Node[T]]
```

**Why generic?**
- `+`, `-`, `*`, `/` work on `int`
- `and`, `or` work on `bool`
- `==`, `!=` work on both
- One class handles all cases with type safety

### Functions

```python
class FnDef(Node[Any], tag="fndef"):
    """
    Function definition: fn(param) => body

    Returns a closure when evaluated.
    """
    param: str              # Parameter name
    body: Ref[Node[Any]]    # Function body

class FnCall(Node[Any], tag="fncall"):
    """Function call: func(arg)"""
    func: Ref[Node[Any]]    # Function to call
    arg: Ref[Node[Any]]     # Argument

class Let(Node[Any], tag="let"):
    """
    Let binding: let name = value in body

    Introduces a new variable in the body's scope.
    """
    name: str               # Variable name
    value: Ref[Node[Any]]   # Value to bind
    body: Ref[Node[Any]]    # Body expression
```

---

## Step 2: Build the Evaluator (12 min)

### Evaluator with Environment

```python
class Evaluator(Interpreter[dict[str, Any], Any]):
    """
    Evaluate expressions.

    Context: dict[str, Any] - variable environment
    Returns: Any - result of evaluation
    """

    def eval(self, node: Node[Any]) -> Any:
        match node:
            # Literals
            case IntLit(value=v):
                return v

            case BoolLit(value=v):
                return v

            # Variables
            case Var(name=n):
                if n not in self.ctx:
                    raise NameError(f"Undefined variable: {n}")
                return self.ctx[n]

            # Binary operations
            case BinOp(op=op, left=l, right=r):
                left_val = self.eval(self.resolve(l))
                right_val = self.eval(self.resolve(r))

                match op:
                    case "+":
                        return left_val + right_val
                    case "-":
                        return left_val - right_val
                    case "*":
                        return left_val * right_val
                    case "/":
                        if right_val == 0:
                            raise ZeroDivisionError("Division by zero")
                        return left_val // right_val  # Integer division
                    case "==":
                        return left_val == right_val
                    case "!=":
                        return left_val != right_val
                    case "<":
                        return left_val < right_val
                    case ">":
                        return left_val > right_val
                    case "and":
                        return left_val and right_val
                    case "or":
                        return left_val or right_val

            # Function definition - return a closure
            case FnDef(param=p, body=b):
                # Capture current environment
                captured_env = self.ctx.copy()

                # Return a closure
                def closure(arg):
                    # Evaluate body with parameter bound
                    old_ctx = self.ctx
                    self.ctx = {**captured_env, p: arg}
                    result = self.eval(self.resolve(b))
                    self.ctx = old_ctx
                    return result

                return closure

            # Function call
            case FnCall(func=f, arg=a):
                func_val = self.eval(self.resolve(f))
                arg_val = self.eval(self.resolve(a))

                if not callable(func_val):
                    raise TypeError(f"Cannot call non-function: {func_val}")

                return func_val(arg_val)

            # Let binding
            case Let(name=n, value=v, body=b):
                val = self.eval(self.resolve(v))

                # Evaluate body with new binding
                old_ctx = self.ctx
                self.ctx = {**self.ctx, n: val}
                result = self.eval(self.resolve(b))
                self.ctx = old_ctx

                return result

            case _:
                raise NotImplementedError(f"Unknown node: {type(node)}")
```

### Test It: Simple Expression

```python
# Build: let x = 5 in x * 2
ast = AST(
    root="result",
    nodes={
        "five": IntLit(value=5),
        "two": IntLit(value=2),
        "x_ref": Var(name="x"),
        "mult": BinOp[int](op="*", left=Ref(id="x_ref"), right=Ref(id="two")),
        "result": Let(name="x", value=Ref(id="five"), body=Ref(id="mult"))
    }
)

evaluator = Evaluator(ast, {})
result = evaluator.run()
print(result)  # 10
```

### Test It: Function Definition

```python
# Build: let double = fn(x) => x * 2 in double(5)
ast = AST(
    root="result",
    nodes={
        # Function body: x * 2
        "x_ref": Var(name="x"),
        "two": IntLit(value=2),
        "body": BinOp[int](op="*", left=Ref(id="x_ref"), right=Ref(id="two")),

        # Function: fn(x) => x * 2
        "double": FnDef(param="x", body=Ref(id="body")),

        # Call: double(5)
        "five": IntLit(value=5),
        "call": FnCall(func=Ref(id="double_ref"), arg=Ref(id="five")),

        # Let: let double = ... in call
        "double_ref": Var(name="double"),
        "result": Let(name="double", value=Ref(id="double"), body=Ref(id="call"))
    }
)

evaluator = Evaluator(ast, {})
result = evaluator.run()
print(result)  # 10
```

---

## Step 3: Add Type Checking (12 min)

### Type Checker Interpreter

```python
class TypeChecker(Interpreter[dict[str, str], str]):
    """
    Check types without evaluating.

    Context: dict[str, str] - variable types
    Returns: str - type of expression ("int" or "bool")
    """

    def eval(self, node: Node[Any]) -> str:
        match node:
            # Literals have obvious types
            case IntLit():
                return "int"

            case BoolLit():
                return "bool"

            # Variables look up in environment
            case Var(name=n):
                if n not in self.ctx:
                    raise TypeError(f"Undefined variable: {n}")
                return self.ctx[n]

            # Binary operations
            case BinOp(op=op, left=l, right=r):
                left_type = self.eval(self.resolve(l))
                right_type = self.eval(self.resolve(r))

                # Both sides must match
                if left_type != right_type:
                    raise TypeError(
                        f"Type mismatch: {left_type} != {right_type}"
                    )

                # Determine result type based on operator
                match op:
                    case "+" | "-" | "*" | "/":
                        if left_type != "int":
                            raise TypeError(f"Arithmetic on non-int: {left_type}")
                        return "int"

                    case "==" | "!=" | "<" | ">":
                        # Comparison returns bool
                        return "bool"

                    case "and" | "or":
                        if left_type != "bool":
                            raise TypeError(f"Logical op on non-bool: {left_type}")
                        return "bool"

            # Function definition
            case FnDef(param=p, body=b):
                # We'd need more sophisticated type inference here
                # For now, just check the body is well-typed
                # Assume parameter is int (simplified)
                old_ctx = self.ctx
                self.ctx = {**self.ctx, p: "int"}
                body_type = self.eval(self.resolve(b))
                self.ctx = old_ctx

                # Return a function type (simplified as string)
                return f"int -> {body_type}"

            # Function call
            case FnCall(func=f, arg=a):
                func_type = self.eval(self.resolve(f))
                arg_type = self.eval(self.resolve(a))

                # Parse function type (simplified)
                if " -> " not in func_type:
                    raise TypeError(f"Not a function: {func_type}")

                param_type, return_type = func_type.split(" -> ")

                if param_type != arg_type:
                    raise TypeError(
                        f"Argument type mismatch: expected {param_type}, got {arg_type}"
                    )

                return return_type

            # Let binding
            case Let(name=n, value=v, body=b):
                value_type = self.eval(self.resolve(v))

                old_ctx = self.ctx
                self.ctx = {**self.ctx, n: value_type}
                body_type = self.eval(self.resolve(b))
                self.ctx = old_ctx

                return body_type

            case _:
                raise NotImplementedError(f"Unknown node: {type(node)}")
```

### Use Type Checking

```python
# Type check before evaluation
type_checker = TypeChecker(ast, {})
try:
    expr_type = type_checker.run()
    print(f"Expression type: {expr_type}")

    # Only evaluate if type-safe
    evaluator = Evaluator(ast, {})
    result = evaluator.run()
    print(f"Result: {result}")
except TypeError as e:
    print(f"Type error: {e}")
```

### Example: Catch Type Errors

```python
# Build: 5 + true (type error!)
ast = AST(
    root="result",
    nodes={
        "five": IntLit(value=5),
        "true": BoolLit(value=True),
        "result": BinOp[???](op="+", left=Ref(id="five"), right=Ref(id="true"))
        #             ^^^ What type? int or bool? Neither works!
    }
)

# Type checking catches the error
type_checker = TypeChecker(ast, {})
try:
    type_checker.run()
except TypeError as e:
    print(f"Caught type error: {e}")
    # Output: Caught type error: Type mismatch: int != bool
```

---

## Step 4: Complete Example (6 min)

### The Full Pipeline

```python
# Build: let double = fn(x) => x * 2 in double(5) + 3
ast = AST(
    root="result",
    nodes={
        # Function body: x * 2
        "x": Var(name="x"),
        "two": IntLit(value=2),
        "double_body": BinOp[int](op="*", left=Ref(id="x"), right=Ref(id="two")),

        # Function definition
        "double_fn": FnDef(param="x", body=Ref(id="double_body")),

        # Call double(5)
        "five": IntLit(value=5),
        "double_var": Var(name="double"),
        "call": FnCall(func=Ref(id="double_var"), arg=Ref(id="five")),

        # Add 3
        "three": IntLit(value=3),
        "add": BinOp[int](op="+", left=Ref(id="call"), right=Ref(id="three")),

        # Let binding
        "result": Let(name="double", value=Ref(id="double_fn"), body=Ref(id="add"))
    }
)

# 1. Type check
print("Type checking...")
type_checker = TypeChecker(ast, {})
expr_type = type_checker.run()
print(f"✓ Expression type: {expr_type}")

# 2. Serialize
print("\nSerializing...")
json_str = ast.to_json()
print(f"✓ Serialized to {len(json_str)} bytes")

# 3. Deserialize
print("\nDeserializing...")
restored_ast = AST.from_json(json_str)
print("✓ Deserialized successfully")

# 4. Evaluate
print("\nEvaluating...")
evaluator = Evaluator(restored_ast, {})
result = evaluator.run()
print(f"✓ Result: {result}")
# Output: 13
```

---

## Extensions

### 1. Add More Types

```python
class StrLit(Node[str], tag="strlit"):
    value: str

class Concat[T: str](Node[T], tag="concat"):
    left: Ref[Node[T]]
    right: Ref[Node[T]]
```

### 2. Add If/Else

```python
class IfElse[T](Node[T], tag="ifelse"):
    condition: Ref[Node[bool]]
    then_branch: Ref[Node[T]]
    else_branch: Ref[Node[T]]
```

### 3. Add Lists

```python
class ListLit[T](Node[list[T]], tag="listlit"):
    elements: list[Ref[Node[T]]]

class Map[E, R](Node[list[R]], tag="map"):
    input: Ref[Node[list[E]]]
    func: Ref[Node[Any]]  # Function E -> R
```

### 4. Better Type Inference

Implement proper Hindley-Milner type inference for automatic type deduction.

---

## Key Takeaways

✓ **Generic nodes** (`BinOp[T]`) enable polymorphic operations
✓ **Type parameters** catch errors at AST construction time
✓ **Multiple interpreters** (type checker, evaluator) analyze from different angles
✓ **Closures** enable first-class functions
✓ **Serialization** preserves entire language expressions

---

## Next Steps

- Try `SQL_QUERY_BUILDER.md` for a concrete, real-world DSL
- Extend this language with your own features
- Build a parser to convert text → AST
