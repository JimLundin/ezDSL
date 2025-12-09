"""
Adapting an Existing AST
=========================

Shows how to convert an external AST (Python's ast module) to nanoDSL.
This pattern applies to ANY existing AST: tree-sitter, ANTLR, proprietary parsers, etc.
"""

import ast as python_ast
from typing import Literal, Any
from nanodsl import Node, Ref, AST, Interpreter


# ============================================================================
# Define nanoDSL Nodes Matching Python AST
# ============================================================================

class Constant(Node[Any], tag="py_const"):
    """Maps to python_ast.Constant"""
    value: Any  # int, float, str, bool, None


class Name(Node[Any], tag="py_name"):
    """Maps to python_ast.Name"""
    id: str


class BinOp(Node[Any], tag="py_binop"):
    """Maps to python_ast.BinOp"""
    left: Ref[Node[Any]]
    op: Literal["Add", "Sub", "Mult", "Div"]
    right: Ref[Node[Any]]


# ============================================================================
# Converter: Python AST → nanoDSL
# ============================================================================

class PythonASTConverter:
    """Converts Python AST to nanoDSL AST."""

    def __init__(self):
        self.nodes: dict[str, Node[Any]] = {}
        self.counter = 0

    def convert(self, py_node: python_ast.AST) -> AST:
        """Convert Python AST to nanoDSL AST."""
        root_id = self._convert_node(py_node)
        return AST(root=root_id, nodes=self.nodes)

    def _convert_node(self, py_node: python_ast.AST) -> str:
        """Convert a single Python AST node."""
        match py_node:
            case python_ast.Constant(value=v):
                node_id = f"const_{self.counter}"
                self.counter += 1
                self.nodes[node_id] = Constant(value=v)
                return node_id

            case python_ast.Name(id=name):
                node_id = f"name_{self.counter}"
                self.counter += 1
                self.nodes[node_id] = Name(id=name)
                return node_id

            case python_ast.BinOp(left=left, op=op, right=right):
                left_id = self._convert_node(left)
                right_id = self._convert_node(right)
                op_name = type(op).__name__  # "Add", "Sub", etc.

                node_id = f"binop_{self.counter}"
                self.counter += 1
                self.nodes[node_id] = BinOp(
                    left=Ref(id=left_id),
                    op=op_name,
                    right=Ref(id=right_id)
                )
                return node_id

            case _:
                raise NotImplementedError(
                    f"Not implemented: {type(py_node).__name__}"
                )


# ============================================================================
# Interpreter for nanoDSL Python AST
# ============================================================================

class PythonASTInterpreter(Interpreter[dict[str, Any], Any]):
    """Evaluates nanoDSL Python AST nodes."""

    def eval(self, node: Node[Any]) -> Any:
        match node:
            case Constant(value=v):
                return v

            case Name(id=name):
                if name not in self.ctx:
                    raise NameError(f"Name '{name}' is not defined")
                return self.ctx[name]

            case BinOp(left=l, op=op, right=r):
                left_val = self.eval(self.resolve(l))
                right_val = self.eval(self.resolve(r))

                match op:
                    case "Add":
                        return left_val + right_val
                    case "Sub":
                        return left_val - right_val
                    case "Mult":
                        return left_val * right_val
                    case "Div":
                        return left_val / right_val

            case _:
                raise NotImplementedError(f"Unknown node: {type(node)}")


# ============================================================================
# Example
# ============================================================================

def main():
    """Convert and evaluate: (a + b) * 2"""

    python_code = "(a + b) * 2"
    print(f"Python code: {python_code}\n")

    # Parse with Python's ast module
    py_ast = python_ast.parse(python_code, mode="eval")

    # Convert to nanoDSL
    converter = PythonASTConverter()
    nano_ast = converter.convert(py_ast.body)

    print(f"Created {len(nano_ast.nodes)} nanoDSL nodes\n")

    # Serialize to JSON
    json_str = nano_ast.to_json()
    print("nanoDSL AST as JSON:")
    print(json_str)
    print()

    # Deserialize and evaluate
    restored_ast = AST.from_json(json_str)
    env = {"a": 3, "b": 7}
    interpreter = PythonASTInterpreter(restored_ast, env)
    result = interpreter.run()

    print(f"Environment: {env}")
    print(f"Result: {result}")
    print(f"Expected: {eval(python_code, env)}")


if __name__ == "__main__":
    main()
