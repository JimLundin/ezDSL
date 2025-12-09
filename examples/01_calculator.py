"""
Calculator DSL Example
======================

A simple mathematical expression evaluator demonstrating:
- Node definition
- The Interpreter base class
- Reference-based ASTs with shared subexpressions
"""

from typing import Literal
from nanodsl import Node, Ref, AST, Interpreter


# ============================================================================
# Define Nodes
# ============================================================================

class Const(Node[float], tag="calc_const"):
    """A constant numeric value."""
    value: float


class Var(Node[float], tag="calc_var"):
    """A variable reference."""
    name: str


class BinOp(Node[float], tag="calc_binop"):
    """Binary operation: +, -, *, /"""
    op: Literal["+", "-", "*", "/"]
    left: Ref[Node[float]]
    right: Ref[Node[float]]


# ============================================================================
# Implement Interpreter
# ============================================================================

class Calculator(Interpreter[dict[str, float], float]):
    """
    Evaluates calculator expressions.

    Context: dict[str, float] - variable environment
    Returns: float - result of evaluation
    """

    def eval(self, node: Node[float]) -> float:
        """Evaluate a node using pattern matching."""
        match node:
            case Const(value=v):
                return v

            case Var(name=n):
                if n not in self.ctx:
                    raise ValueError(f"Undefined variable: {n}")
                return self.ctx[n]

            case BinOp(op="+", left=l, right=r):
                return self.eval(self.resolve(l)) + self.eval(self.resolve(r))

            case BinOp(op="-", left=l, right=r):
                return self.eval(self.resolve(l)) - self.eval(self.resolve(r))

            case BinOp(op="*", left=l, right=r):
                return self.eval(self.resolve(l)) * self.eval(self.resolve(r))

            case BinOp(op="/", left=l, right=r):
                right_val = self.eval(self.resolve(r))
                if right_val == 0:
                    raise ZeroDivisionError("Division by zero")
                return self.eval(self.resolve(l)) / right_val

            case _:
                raise NotImplementedError(f"Unknown node: {type(node)}")


# ============================================================================
# Example: Shared Subexpressions
# ============================================================================

def main():
    """
    Build and evaluate: (x + y) * (x + y)

    The subexpression (x + y) is shared - computed once, reused twice.
    """

    # Build AST with explicit node IDs
    ast = AST(
        root="result",
        nodes={
            "x": Const(value=3.0),
            "y": Const(value=4.0),
            "sum": BinOp(op="+", left=Ref(id="x"), right=Ref(id="y")),
            "result": BinOp(op="*", left=Ref(id="sum"), right=Ref(id="sum")),
        }
    )

    # Serialize to JSON
    print("AST as JSON:")
    print(ast.to_json())
    print()

    # Evaluate
    calculator = Calculator(ast, {})
    result = calculator.run()

    print(f"Expression: (x + y) * (x + y) where x=3, y=4")
    print(f"Result: {result}")
    print(f"Expected: (3 + 4) * (3 + 4) = 49")
    print()

    # With variables
    ast2 = AST(
        root="expr",
        nodes={
            "a": Var(name="a"),
            "b": Const(value=2.0),
            "expr": BinOp(op="*", left=Ref(id="a"), right=Ref(id="b")),
        }
    )

    calculator2 = Calculator(ast2, {"a": 5.0})
    result2 = calculator2.run()

    print(f"Expression: a * 2 where a=5")
    print(f"Result: {result2}")


if __name__ == "__main__":
    main()
