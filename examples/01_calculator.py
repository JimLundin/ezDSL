"""
Calculator DSL Example
======================

This example demonstrates how to build a simple mathematical expression evaluator
using nanoDSL. It covers:

1. Defining AST nodes for a DSL
2. Building expression trees
3. Using the Interpreter base class for evaluation
4. Working with both inline nodes and references
5. Serialization and deserialization

The calculator supports:
- Constants and variables
- Binary operations: +, -, *, /
- Unary operations: negation, absolute value
"""

from typing import Literal

from nanodsl import AST, Node, Ref, Interpreter, to_json, from_json


# ============================================================================
# Step 1: Define AST Nodes
# ============================================================================
# Each node is parameterized by the type it produces (Node[T]).
# Tags are automatically derived from class names unless explicitly specified.


class Const(Node[float], tag="calc_const"):
    """A constant numeric value."""

    value: float


class Var(Node[float], tag="calc_var"):
    """A variable reference (e.g., 'x', 'y')."""

    name: str


class BinOp(Node[float], tag="calc_binop"):
    """Binary operation: +, -, *, /"""

    op: Literal["+", "-", "*", "/"]
    left: Node[float]  # Direct nesting - contains the actual node
    right: Node[float]


class UnaryOp(Node[float], tag="calc_unary"):
    """Unary operation: negation (-) or absolute value (abs)."""

    op: Literal["-", "abs"]
    operand: Node[float]


# ============================================================================
# Step 2: Build Expression Trees (Inline Nodes)
# ============================================================================
# When using direct Node[T] fields, we build tree structures with inline nodes.


def example_inline_tree():
    """Build an expression tree using inline nodes: abs(x + 2) * 3"""

    expr = BinOp(
        op="*",
        left=UnaryOp(
            op="abs",
            operand=BinOp(op="+", left=Var(name="x"), right=Const(value=2.0)),
        ),
        right=Const(value=3.0),
    )

    # Serialize to JSON
    json_str = to_json(expr)
    print("Expression as JSON:")
    print(json_str)
    print()

    # Deserialize back
    restored = from_json(json_str)
    assert restored == expr
    print(f"Successfully round-tripped: {restored == expr}")
    print()

    return expr


# ============================================================================
# Step 3: Implement an Interpreter (Inline Evaluation)
# ============================================================================
# The Interpreter base class provides a clean pattern for AST evaluation.


class CalculatorInterpreter(Interpreter[dict[str, float], float]):
    """
    Evaluates calculator expressions.

    Type parameters:
    - Ctx = dict[str, float]: Environment mapping variable names to values
    - R = float: Return type of evaluation
    """

    def eval(self, node: Node[float]) -> float:
        """
        Evaluate a single node.

        Uses pattern matching (Python 3.10+) for clean dispatch.
        """
        match node:
            case Const(value=v):
                return v

            case Var(name=n):
                if n not in self.ctx:
                    raise ValueError(f"Undefined variable: {n}")
                return self.ctx[n]

            case BinOp(op="+", left=l, right=r):
                return self.eval(l) + self.eval(r)

            case BinOp(op="-", left=l, right=r):
                return self.eval(l) - self.eval(r)

            case BinOp(op="*", left=l, right=r):
                return self.eval(l) * self.eval(r)

            case BinOp(op="/", left=l, right=r):
                right_val = self.eval(r)
                if right_val == 0:
                    raise ZeroDivisionError("Division by zero")
                return self.eval(l) / right_val

            case UnaryOp(op="-", operand=o):
                return -self.eval(o)

            case UnaryOp(op="abs", operand=o):
                return abs(self.eval(o))

            case _:
                raise NotImplementedError(f"Unknown node type: {type(node)}")


def example_inline_evaluation():
    """Evaluate an inline expression tree."""

    # Build expression: abs(x + 2) * 3
    expr = BinOp(
        op="*",
        left=UnaryOp(
            op="abs",
            operand=BinOp(op="+", left=Var(name="x"), right=Const(value=2.0)),
        ),
        right=Const(value=3.0),
    )

    # Create interpreter with environment
    env = {"x": -5.0}
    interpreter = CalculatorInterpreter(None, env)  # No AST needed for inline trees

    # Evaluate (note: for inline trees, we call eval directly)
    result = interpreter.eval(expr)

    print(f"Expression: abs(x + 2) * 3")
    print(f"Environment: x = {env['x']}")
    print(f"Result: {result}")
    print(f"Verification: abs(-5.0 + 2.0) * 3 = abs(-3.0) * 3 = 9.0")
    print()

    assert result == 9.0


# ============================================================================
# Step 4: Working with References (Graph Structures)
# ============================================================================
# For DAG structures with shared nodes, use Ref[Node[T]] and AST container.


# First, redefine nodes to use references
class RefBinOp(Node[float], tag="calc_ref_binop"):
    """Binary operation with references instead of inline nodes."""

    op: Literal["+", "-", "*", "/"]
    left: Ref[Node[float]]  # Reference to a node by ID
    right: Ref[Node[float]]


class RefUnaryOp(Node[float], tag="calc_ref_unary"):
    """Unary operation with references."""

    op: Literal["-", "abs"]
    operand: Ref[Node[float]]


def example_reference_based_ast():
    """Build an AST with references to enable node sharing."""

    # Build expression with shared subexpression:
    # result = (x + y) * (x + y)
    # The subexpression (x + y) is computed once and reused

    ast = AST(
        root="result",
        nodes={
            "x": Const(value=3.0),
            "y": Const(value=4.0),
            "sum": RefBinOp(
                op="+", left=Ref(id="x"), right=Ref(id="y")
            ),  # x + y
            "result": RefBinOp(
                op="*", left=Ref(id="sum"), right=Ref(id="sum")
            ),  # (x+y) * (x+y)
        },
    )

    # Serialize entire AST
    json_str = ast.to_json()
    print("AST with shared nodes as JSON:")
    print(json_str)
    print()

    # Deserialize
    restored_ast = AST.from_json(json_str)
    assert restored_ast == ast
    print(f"Successfully round-tripped AST: {restored_ast == ast}")
    print()

    return ast


class RefCalculatorInterpreter(Interpreter[dict[str, float], float]):
    """
    Evaluates calculator expressions with references.

    This interpreter handles both inline nodes (Const, Var) and
    reference-based nodes (RefBinOp, RefUnaryOp).
    """

    def eval(self, node: Node[float]) -> float:
        """Evaluate a single node, resolving references as needed."""
        match node:
            case Const(value=v):
                return v

            case Var(name=n):
                if n not in self.ctx:
                    raise ValueError(f"Undefined variable: {n}")
                return self.ctx[n]

            case RefBinOp(op="+", left=l, right=r):
                # Resolve references to get actual nodes, then evaluate
                return self.eval(self.resolve(l)) + self.eval(self.resolve(r))

            case RefBinOp(op="-", left=l, right=r):
                return self.eval(self.resolve(l)) - self.eval(self.resolve(r))

            case RefBinOp(op="*", left=l, right=r):
                return self.eval(self.resolve(l)) * self.eval(self.resolve(r))

            case RefBinOp(op="/", left=l, right=r):
                right_val = self.eval(self.resolve(r))
                if right_val == 0:
                    raise ZeroDivisionError("Division by zero")
                return self.eval(self.resolve(l)) / right_val

            case RefUnaryOp(op="-", operand=o):
                return -self.eval(self.resolve(o))

            case RefUnaryOp(op="abs", operand=o):
                return abs(self.eval(self.resolve(o)))

            case _:
                raise NotImplementedError(f"Unknown node type: {type(node)}")


def example_reference_evaluation():
    """Evaluate a reference-based AST."""

    # Build AST: result = (x + y) * (x + y) where sum is shared
    ast = AST(
        root="result",
        nodes={
            "x": Const(value=3.0),
            "y": Const(value=4.0),
            "sum": RefBinOp(op="+", left=Ref(id="x"), right=Ref(id="y")),
            "result": RefBinOp(op="*", left=Ref(id="sum"), right=Ref(id="sum")),
        },
    )

    # Create interpreter (empty environment since we use constants)
    interpreter = RefCalculatorInterpreter(ast, {})

    # Evaluate from root
    result = interpreter.run()

    print(f"Expression: (x + y) * (x + y) where x=3.0, y=4.0")
    print(f"Result: {result}")
    print(f"Verification: (3.0 + 4.0) * (3.0 + 4.0) = 7.0 * 7.0 = 49.0")
    print()

    assert result == 49.0


# ============================================================================
# Step 5: Memoization Pattern
# ============================================================================
# For expensive computations or DAGs, memoize node evaluation results.


class MemoizedCalculator(Interpreter[dict[str, float], float]):
    """
    Calculator interpreter with memoization.

    Caches evaluation results by node ID to avoid recomputing shared subexpressions.
    """

    def __init__(self, ast: AST, ctx: dict[str, float]) -> None:
        super().__init__(ast, ctx)
        self._cache: dict[str, float] = {}
        self._eval_count: dict[str, int] = {}  # Track evaluation calls per node

    def eval_ref(self, ref: Ref[Node[float]]) -> float:
        """Evaluate a reference with memoization."""
        if ref.id not in self._cache:
            # Track how many times we evaluate each node
            self._eval_count[ref.id] = self._eval_count.get(ref.id, 0) + 1
            self._cache[ref.id] = self.eval(self.resolve(ref))
        return self._cache[ref.id]

    def eval(self, node: Node[float]) -> float:
        """Evaluate a node."""
        match node:
            case Const(value=v):
                return v

            case Var(name=n):
                if n not in self.ctx:
                    raise ValueError(f"Undefined variable: {n}")
                return self.ctx[n]

            case RefBinOp(op="+", left=l, right=r):
                return self.eval_ref(l) + self.eval_ref(r)

            case RefBinOp(op="-", left=l, right=r):
                return self.eval_ref(l) - self.eval_ref(r)

            case RefBinOp(op="*", left=l, right=r):
                return self.eval_ref(l) * self.eval_ref(r)

            case RefBinOp(op="/", left=l, right=r):
                right_val = self.eval_ref(r)
                if right_val == 0:
                    raise ZeroDivisionError("Division by zero")
                return self.eval_ref(l) / right_val

            case RefUnaryOp(op="-", operand=o):
                return -self.eval_ref(o)

            case RefUnaryOp(op="abs", operand=o):
                return abs(self.eval_ref(o))

            case _:
                raise NotImplementedError(f"Unknown node type: {type(node)}")


def example_memoization():
    """Demonstrate memoization for shared subexpressions."""

    # Build a diamond pattern where 'sum' is referenced multiple times:
    #     result = (sum * 2) + (sum * 3)
    #     sum = x + y
    ast = AST(
        root="result",
        nodes={
            "x": Const(value=5.0),
            "y": Const(value=3.0),
            "sum": RefBinOp(op="+", left=Ref(id="x"), right=Ref(id="y")),
            "left_mult": RefBinOp(op="*", left=Ref(id="sum"), right=Ref(id="two")),
            "right_mult": RefBinOp(op="*", left=Ref(id="sum"), right=Ref(id="three")),
            "two": Const(value=2.0),
            "three": Const(value=3.0),
            "result": RefBinOp(
                op="+", left=Ref(id="left_mult"), right=Ref(id="right_mult")
            ),
        },
    )

    # Evaluate with memoization
    interpreter = MemoizedCalculator(ast, {})
    result = interpreter.run()

    print("Expression: (sum * 2) + (sum * 3) where sum = x + y, x=5.0, y=3.0")
    print(f"Result: {result}")
    print(f"Verification: (8.0 * 2) + (8.0 * 3) = 16.0 + 24.0 = 40.0")
    print()
    print("Node evaluation counts:")
    for node_id, count in interpreter._eval_count.items():
        print(f"  {node_id}: {count} time(s)")
    print()
    print(
        "Note: 'sum' is only evaluated once despite being referenced twice (memoization)"
    )
    print()

    assert result == 40.0
    assert interpreter._eval_count["sum"] == 1  # Only evaluated once!


# ============================================================================
# Main: Run All Examples
# ============================================================================


def main():
    """Run all calculator examples."""

    print("=" * 80)
    print("Calculator DSL Example")
    print("=" * 80)
    print()

    print("--- Example 1: Inline Expression Tree ---")
    example_inline_tree()
    print()

    print("--- Example 2: Inline Evaluation ---")
    example_inline_evaluation()
    print()

    print("--- Example 3: Reference-Based AST ---")
    example_reference_based_ast()
    print()

    print("--- Example 4: Reference-Based Evaluation ---")
    example_reference_evaluation()
    print()

    print("--- Example 5: Memoization ---")
    example_memoization()
    print()

    print("All examples completed successfully! ✓")


if __name__ == "__main__":
    main()
