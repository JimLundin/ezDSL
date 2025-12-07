"""
Adapting an Existing AST Example
=================================

This example demonstrates how to adapt an existing AST from another system
(like Python's ast module, tree-sitter, or a proprietary parser) to use nanoDSL.

It covers:

1. Mapping external AST nodes to nanoDSL nodes
2. Converting between representations
3. Preserving type information
4. Building interpreters for the adapted AST
5. Round-tripping: external → nanoDSL → external

This example uses Python's built-in `ast` module as the external AST,
but the patterns apply to any existing AST system.
"""

import ast as python_ast
from typing import Literal, Any

from nanodsl import Node, Ref, AST, Interpreter


# ============================================================================
# Step 1: Define nanoDSL Nodes Matching Python AST
# ============================================================================
# We create nanoDSL nodes that mirror Python's AST structure.
# Each node is parameterized by what it produces when evaluated.


# --- Expressions (produce values) ---


class Constant(Node[Any], tag="py_const"):
    """
    A constant value (number, string, bool, None).

    Maps to: python_ast.Constant
    """

    value: Any  # int, float, str, bool, None
    # Note: In production, you might want separate nodes for each type


class Name(Node[Any], tag="py_name"):
    """
    A variable reference.

    Maps to: python_ast.Name
    """

    id: str  # Variable name


class BinOp(Node[Any], tag="py_binop"):
    """
    Binary operation (e.g., a + b, x * y).

    Maps to: python_ast.BinOp
    """

    left: Ref[Node[Any]]
    op: Literal["Add", "Sub", "Mult", "Div", "Mod", "Pow"]
    right: Ref[Node[Any]]


class UnaryOp(Node[Any], tag="py_unaryop"):
    """
    Unary operation (e.g., -x, not y).

    Maps to: python_ast.UnaryOp
    """

    op: Literal["UAdd", "USub", "Not"]
    operand: Ref[Node[Any]]


class Compare(Node[bool], tag="py_compare"):
    """
    Comparison operation (e.g., x < y, a == b).

    Maps to: python_ast.Compare
    """

    left: Ref[Node[Any]]
    op: Literal["Eq", "NotEq", "Lt", "LtE", "Gt", "GtE"]
    comparator: Ref[Node[Any]]  # Simplified: Python AST supports multiple comparisons


# --- Statements (produce None, cause side effects) ---


class Assign(Node[None], tag="py_assign"):
    """
    Assignment statement (e.g., x = 5).

    Maps to: python_ast.Assign
    """

    target: str  # Variable name (simplified)
    value: Ref[Node[Any]]


class Expr(Node[None], tag="py_expr"):
    """
    Expression statement (evaluate expression, discard result).

    Maps to: python_ast.Expr
    """

    value: Ref[Node[Any]]


class Module(Node[None], tag="py_module"):
    """
    Top-level module containing statements.

    Maps to: python_ast.Module
    """

    body: list[Ref[Node[None]]]  # List of statements


# ============================================================================
# Step 2: Converter - Python AST → nanoDSL
# ============================================================================


class PythonASTConverter:
    """
    Converts Python AST nodes to nanoDSL nodes.

    This converter:
    1. Traverses the Python AST
    2. Creates corresponding nanoDSL nodes
    3. Builds an AST container with references
    4. Preserves structure and semantics
    """

    def __init__(self):
        self.nodes: dict[str, Node[Any]] = {}
        self.node_counter = 0

    def _make_id(self, prefix: str = "node") -> str:
        """Generate a unique node ID."""
        self.node_counter += 1
        return f"{prefix}_{self.node_counter}"

    def convert(self, py_node: python_ast.AST) -> AST:
        """
        Convert a Python AST to nanoDSL AST.

        Args:
            py_node: Root node of Python AST (typically ast.Module)

        Returns:
            AST container with nanoDSL nodes
        """
        root_id = self._convert_node(py_node)
        return AST(root=root_id, nodes=self.nodes)

    def _convert_node(self, py_node: python_ast.AST) -> str:
        """
        Convert a single Python AST node to nanoDSL.

        Returns:
            Node ID in the AST
        """
        match py_node:
            # --- Constants ---
            case python_ast.Constant(value=v):
                node_id = self._make_id("const")
                self.nodes[node_id] = Constant(value=v)
                return node_id

            # --- Variables ---
            case python_ast.Name(id=name):
                node_id = self._make_id("name")
                self.nodes[node_id] = Name(id=name)
                return node_id

            # --- Binary Operations ---
            case python_ast.BinOp(left=left, op=op, right=right):
                left_id = self._convert_node(left)
                right_id = self._convert_node(right)

                # Map Python op to our literal
                op_name = type(op).__name__  # e.g., "Add", "Sub", "Mult"

                node_id = self._make_id("binop")
                self.nodes[node_id] = BinOp(
                    left=Ref(id=left_id), op=op_name, right=Ref(id=right_id)
                )
                return node_id

            # --- Unary Operations ---
            case python_ast.UnaryOp(op=op, operand=operand):
                operand_id = self._convert_node(operand)
                op_name = type(op).__name__  # e.g., "UAdd", "USub"

                node_id = self._make_id("unaryop")
                self.nodes[node_id] = UnaryOp(op=op_name, operand=Ref(id=operand_id))
                return node_id

            # --- Comparisons ---
            case python_ast.Compare(left=left, ops=[op], comparators=[comp]):
                # Simplified: handle single comparison
                left_id = self._convert_node(left)
                comp_id = self._convert_node(comp)
                op_name = type(op).__name__  # e.g., "Lt", "Eq"

                node_id = self._make_id("compare")
                self.nodes[node_id] = Compare(
                    left=Ref(id=left_id), op=op_name, comparator=Ref(id=comp_id)
                )
                return node_id

            # --- Statements ---
            case python_ast.Assign(targets=[python_ast.Name(id=target)], value=value):
                # Simplified: single target
                value_id = self._convert_node(value)

                node_id = self._make_id("assign")
                self.nodes[node_id] = Assign(target=target, value=Ref(id=value_id))
                return node_id

            case python_ast.Expr(value=value):
                value_id = self._convert_node(value)

                node_id = self._make_id("expr")
                self.nodes[node_id] = Expr(value=Ref(id=value_id))
                return node_id

            case python_ast.Module(body=stmts):
                stmt_ids = [self._convert_node(stmt) for stmt in stmts]

                node_id = self._make_id("module")
                self.nodes[node_id] = Module(body=[Ref(id=sid) for sid in stmt_ids])
                return node_id

            case _:
                raise NotImplementedError(
                    f"Conversion not implemented for: {type(py_node).__name__}"
                )


# ============================================================================
# Step 3: Interpreter for nanoDSL Python AST
# ============================================================================


class PythonASTInterpreter(Interpreter[dict[str, Any], Any]):
    """
    Interprets nanoDSL Python AST nodes.

    This interpreter evaluates Python expressions and executes statements,
    maintaining a variable environment in the context.
    """

    def eval(self, node: Node[Any]) -> Any:
        """Evaluate a nanoDSL Python AST node."""
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
                    case "Mod":
                        return left_val % right_val
                    case "Pow":
                        return left_val**right_val
                    case _:
                        raise NotImplementedError(f"Binary op '{op}' not implemented")

            case UnaryOp(op=op, operand=operand):
                operand_val = self.eval(self.resolve(operand))

                match op:
                    case "UAdd":
                        return +operand_val
                    case "USub":
                        return -operand_val
                    case "Not":
                        return not operand_val
                    case _:
                        raise NotImplementedError(f"Unary op '{op}' not implemented")

            case Compare(left=l, op=op, comparator=c):
                left_val = self.eval(self.resolve(l))
                comp_val = self.eval(self.resolve(c))

                match op:
                    case "Eq":
                        return left_val == comp_val
                    case "NotEq":
                        return left_val != comp_val
                    case "Lt":
                        return left_val < comp_val
                    case "LtE":
                        return left_val <= comp_val
                    case "Gt":
                        return left_val > comp_val
                    case "GtE":
                        return left_val >= comp_val
                    case _:
                        raise NotImplementedError(f"Compare op '{op}' not implemented")

            case Assign(target=target, value=value_ref):
                value = self.eval(self.resolve(value_ref))
                self.ctx[target] = value
                return None

            case Expr(value=value_ref):
                # Evaluate and discard
                self.eval(self.resolve(value_ref))
                return None

            case Module(body=stmts):
                # Execute statements in order
                for stmt_ref in stmts:
                    self.eval(self.resolve(stmt_ref))
                return None

            case _:
                raise NotImplementedError(f"Evaluation not implemented for: {type(node)}")


# ============================================================================
# Step 4: Examples - Converting and Evaluating Python Code
# ============================================================================


def example_simple_expression():
    """Convert and evaluate a simple Python expression."""

    # Python code: 2 + 3 * 4
    python_code = "2 + 3 * 4"

    print("=" * 80)
    print("Example 1: Simple Expression")
    print("=" * 80)
    print(f"Python code: {python_code}")
    print()

    # Parse with Python's ast module
    py_ast = python_ast.parse(python_code, mode="eval")

    # Convert to nanoDSL
    converter = PythonASTConverter()
    nano_ast = converter.convert(py_ast.body)

    # Serialize to JSON
    json_str = nano_ast.to_json()
    print("nanoDSL AST (JSON):")
    print(json_str)
    print()

    # Evaluate
    interpreter = PythonASTInterpreter(nano_ast, {})
    result = interpreter.run()

    print(f"Result: {result}")
    print(f"Expected: {eval(python_code)}")
    print(f"Match: {result == eval(python_code)}")
    print()


def example_with_variables():
    """Convert and evaluate Python code with variables."""

    # Python code: x + y * 2
    python_code = "x + y * 2"

    print("=" * 80)
    print("Example 2: Expression with Variables")
    print("=" * 80)
    print(f"Python code: {python_code}")
    print()

    # Parse
    py_ast = python_ast.parse(python_code, mode="eval")

    # Convert
    converter = PythonASTConverter()
    nano_ast = converter.convert(py_ast.body)

    # Evaluate with environment
    env = {"x": 10, "y": 5}
    interpreter = PythonASTInterpreter(nano_ast, env)
    result = interpreter.run()

    print(f"Environment: {env}")
    print(f"Result: {result}")
    print(f"Expected: {eval(python_code, env)}")
    print(f"Match: {result == eval(python_code, env)}")
    print()


def example_statements():
    """Convert and execute Python statements."""

    # Python code with multiple statements
    python_code = """
x = 5
y = x + 3
result = x * y
"""

    print("=" * 80)
    print("Example 3: Multiple Statements")
    print("=" * 80)
    print("Python code:")
    print(python_code)

    # Parse
    py_ast = python_ast.parse(python_code)

    # Convert
    converter = PythonASTConverter()
    nano_ast = converter.convert(py_ast)

    print("nanoDSL AST structure:")
    print(f"  Root: {nano_ast.root}")
    print(f"  Nodes: {len(nano_ast.nodes)}")
    for node_id, node in nano_ast.nodes.items():
        print(f"    {node_id}: {type(node).__name__}")
    print()

    # Execute
    env = {}
    interpreter = PythonASTInterpreter(nano_ast, env)
    interpreter.run()

    print("Final environment:")
    print(f"  x = {env['x']}")
    print(f"  y = {env['y']}")
    print(f"  result = {env['result']}")
    print()

    # Verify
    exec(python_code, env)
    print(f"Expected result: {env['result']}")
    print()


def example_complex_expression():
    """Convert a more complex expression with multiple operations."""

    # Python code: (a + b) * (c - d) / 2
    python_code = "(a + b) * (c - d) / 2"

    print("=" * 80)
    print("Example 4: Complex Expression")
    print("=" * 80)
    print(f"Python code: {python_code}")
    print()

    # Parse
    py_ast = python_ast.parse(python_code, mode="eval")

    # Convert
    converter = PythonASTConverter()
    nano_ast = converter.convert(py_ast.body)

    print(f"Created {len(nano_ast.nodes)} nanoDSL nodes")
    print()

    # Evaluate
    env = {"a": 10, "b": 5, "c": 20, "d": 8}
    interpreter = PythonASTInterpreter(nano_ast, env)
    result = interpreter.run()

    print(f"Environment: {env}")
    print(f"Result: {result}")
    print(f"Expected: {eval(python_code, env)}")
    print(f"Match: {result == eval(python_code, env)}")
    print()


def example_round_trip():
    """
    Demonstrate round-tripping: Python AST → nanoDSL → JSON → nanoDSL → evaluate.

    This shows that the nanoDSL representation preserves all necessary information.
    """

    python_code = "x * 2 + y"

    print("=" * 80)
    print("Example 5: Round-Trip Serialization")
    print("=" * 80)
    print(f"Python code: {python_code}")
    print()

    # Python AST → nanoDSL
    py_ast = python_ast.parse(python_code, mode="eval")
    converter = PythonASTConverter()
    nano_ast = converter.convert(py_ast.body)

    # nanoDSL → JSON
    json_str = nano_ast.to_json()
    print("Serialized to JSON")
    print()

    # JSON → nanoDSL
    restored_ast = AST.from_json(json_str)
    print("Deserialized from JSON")
    print(f"AST matches: {restored_ast == nano_ast}")
    print()

    # Evaluate restored AST
    env = {"x": 7, "y": 3}
    interpreter = PythonASTInterpreter(restored_ast, env)
    result = interpreter.run()

    print(f"Environment: {env}")
    print(f"Result: {result}")
    print(f"Expected: {eval(python_code, env)}")
    print(f"Match: {result == eval(python_code, env)}")
    print()


# ============================================================================
# Step 5: Key Takeaways and Best Practices
# ============================================================================


def print_best_practices():
    """Print best practices for adapting existing ASTs."""

    print("=" * 80)
    print("Best Practices for Adapting Existing ASTs")
    print("=" * 80)
    print()
    print("1. **Map node types carefully**")
    print("   - Create one nanoDSL node class for each external AST node type")
    print("   - Preserve semantic meaning and type information")
    print()
    print("2. **Use references for child nodes**")
    print("   - Use Ref[Node[T]] for child nodes to enable DAG structures")
    print("   - Store all nodes in an AST container")
    print()
    print("3. **Preserve metadata**")
    print("   - If the external AST has metadata (line numbers, source locations),")
    print("     add these as fields to your nanoDSL nodes")
    print()
    print("4. **Start simple, then extend**")
    print("   - Begin with core node types")
    print("   - Add support for more node types incrementally")
    print("   - Use NotImplementedError for unsupported nodes")
    print()
    print("5. **Test round-tripping**")
    print("   - Verify: external → nanoDSL → JSON → nanoDSL → evaluate")
    print("   - Ensure no information is lost in conversion")
    print()
    print("6. **Consider bidirectional conversion**")
    print("   - If needed, implement nanoDSL → external AST")
    print("   - Useful for code generation or modification")
    print()


# ============================================================================
# Main: Run All Examples
# ============================================================================


def main():
    """Run all examples for adapting existing ASTs."""

    example_simple_expression()
    example_with_variables()
    example_statements()
    example_complex_expression()
    example_round_trip()
    print_best_practices()

    print("All examples completed! ✓")


if __name__ == "__main__":
    main()
