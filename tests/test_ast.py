"""Tests for nanodsl.ast module."""

import json
from typing import Any

import pytest

from nanodsl.ast import AST, Interpreter
from nanodsl.nodes import Node, Ref


class TestASTBasics:
    """Test basic AST functionality."""

    def test_ast_creation(self) -> None:
        """Test creating an AST."""

        class Literal(Node[int], tag="literal_ast"):
            value: int

        nodes = {"node1": Literal(value=42), "node2": Literal(value=100)}
        ast = AST(root="node1", nodes=nodes)

        assert ast.root == "node1"
        assert len(ast.nodes) == 2
        assert "node1" in ast.nodes
        assert "node2" in ast.nodes

    def test_ast_with_empty_nodes(self) -> None:
        """Test creating AST with no nodes."""
        ast = AST(root="", nodes={})

        assert ast.root == ""
        assert len(ast.nodes) == 0

class TestASTResolve:
    """Test AST.resolve() reference resolution."""

    def test_resolve_simple_ref(self) -> None:
        """Test resolving a simple reference."""

        class Number(Node[int], tag="number_resolve"):
            value: int

        node = Number(value=42)
        ast = AST(root="num", nodes={"num": node})

        ref = Ref[Node[int]](id="num")
        resolved = ast.resolve(ref)

        assert resolved is node
        assert resolved.value == 42

    def test_resolve_multiple_refs(self) -> None:
        """Test resolving multiple different references."""

        class Data(Node[str], tag="data_resolve"):
            text: str

        nodes = {
            "first": Data(text="hello"),
            "second": Data(text="world"),
            "third": Data(text="test"),
        }
        ast = AST(root="first", nodes=nodes)

        ref1 = Ref[Node[str]](id="first")
        ref2 = Ref[Node[str]](id="second")
        ref3 = Ref[Node[str]](id="third")

        assert ast.resolve(ref1).text == "hello"
        assert ast.resolve(ref2).text == "world"
        assert ast.resolve(ref3).text == "test"

    def test_resolve_with_shared_nodes(self) -> None:
        """Test resolving refs in a graph with shared nodes."""

        class Const(Node[float], tag="const_shared"):
            value: float

        class Add(Node[float], tag="add_shared"):
            left: Ref[Node[float]]
            right: Ref[Node[float]]

        # Create AST where "x" is shared
        ast = AST(
            root="result",
            nodes={
                "x": Const(value=5.0),
                "y": Const(value=3.0),
                "sum": Add(left=Ref(id="x"), right=Ref(id="y")),
                "result": Add(left=Ref(id="sum"), right=Ref(id="x")),
            },
        )

        # Resolve x from two different contexts
        x_from_sum = ast.resolve(ast.nodes["sum"].left)
        x_from_result = ast.resolve(ast.nodes["result"].right)

        # Should be the same object
        assert x_from_sum is x_from_result
        assert x_from_sum.value == 5.0

    def test_resolve_nonexistent_ref_raises_error(self) -> None:
        """Test that resolving nonexistent ref raises KeyError with helpful message."""

        class Item(Node[int], tag="item_resolve_err"):
            value: int

        ast = AST(root="root", nodes={"root": Item(value=1)})

        ref = Ref[Node[int]](id="nonexistent")

        with pytest.raises(KeyError, match="Node 'nonexistent' not found in AST"):
            ast.resolve(ref)

    def test_resolve_error_lists_available_nodes(self) -> None:
        """Test that error message lists available node IDs."""

        class Item(Node[int], tag="item_resolve_list"):
            value: int

        ast = AST(
            root="a",
            nodes={"a": Item(value=1), "b": Item(value=2), "c": Item(value=3)},
        )

        ref = Ref[Node[int]](id="missing")

        with pytest.raises(KeyError) as exc_info:
            ast.resolve(ref)

        error_msg = str(exc_info.value)
        # Error should mention available IDs
        assert "Available node IDs" in error_msg
        assert "'a'" in error_msg or "a" in error_msg
        assert "'b'" in error_msg or "b" in error_msg
        assert "'c'" in error_msg or "c" in error_msg

    def test_resolve_type_casting(self) -> None:
        """Test that resolve returns correctly typed nodes."""

        class StringNode(Node[str], tag="string_resolve_type"):
            text: str

        node = StringNode(text="hello")
        ast = AST(root="str", nodes={"str": node})

        ref = Ref[Node[str]](id="str")
        resolved = ast.resolve(ref)

        # Should have the correct type at runtime
        assert isinstance(resolved, StringNode)
        assert resolved.text == "hello"


class TestASTSerialization:
    """Test AST serialization methods."""

    def test_to_dict_simple(self) -> None:
        """Test serializing simple AST to dict."""

        class Num(Node[int], tag="num_ast_dict"):
            value: int

        ast = AST(root="n1", nodes={"n1": Num(value=42), "n2": Num(value=100)})

        result = ast.to_dict()

        assert result["root"] == "n1"
        assert "nodes" in result
        assert "n1" in result["nodes"]
        assert "n2" in result["nodes"]
        assert result["nodes"]["n1"]["tag"] == "num_ast_dict"
        assert result["nodes"]["n1"]["value"] == 42

    def test_to_dict_with_refs(self) -> None:
        """Test serializing AST with references."""

        class Val(Node[int], tag="val_ast_dict_ref"):
            value: int

        class Op(Node[int], tag="op_ast_dict_ref"):
            left: Ref[Node[int]]
            right: Ref[Node[int]]

        ast = AST(
            root="result",
            nodes={
                "a": Val(value=1),
                "b": Val(value=2),
                "result": Op(left=Ref(id="a"), right=Ref(id="b")),
            },
        )

        result = ast.to_dict()

        assert result["root"] == "result"
        assert result["nodes"]["result"]["left"]["tag"] == "ref"
        assert result["nodes"]["result"]["left"]["id"] == "a"

    def test_to_json_produces_valid_json(self) -> None:
        """Test that to_json() produces valid JSON."""

        class Simple(Node[str], tag="simple_ast_json"):
            text: str

        ast = AST(root="s", nodes={"s": Simple(text="test")})

        json_str = ast.to_json()

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed["root"] == "s"
        assert "nodes" in parsed

    def test_to_json_is_formatted(self) -> None:
        """Test that to_json() output is formatted."""

        class Item(Node[int], tag="item_ast_json_fmt"):
            value: int

        ast = AST(root="i", nodes={"i": Item(value=1)})

        json_str = ast.to_json()

        # Should contain newlines (formatted)
        assert "\n" in json_str


class TestASTDeserialization:
    """Test AST deserialization methods."""

    def test_from_dict_missing_root_key(self) -> None:
        """Test that from_dict raises error when 'root' key is missing."""
        data = {"nodes": {}}  # Missing 'root' key

        with pytest.raises(KeyError, match="Missing required key 'root'"):
            AST.from_dict(data)

    def test_from_dict_missing_nodes_key(self) -> None:
        """Test that from_dict raises error when 'nodes' key is missing."""
        data = {"root": "test"}  # Missing 'nodes' key

        with pytest.raises(KeyError, match="Missing required key 'nodes'"):
            AST.from_dict(data)

    def test_from_dict_simple(self) -> None:
        """Test deserializing simple AST from dict."""

        class Number(Node[int], tag="number_ast_from_dict"):
            value: int

        data = {
            "root": "n1",
            "nodes": {
                "n1": {"tag": "number_ast_from_dict", "value": 42},
                "n2": {"tag": "number_ast_from_dict", "value": 100},
            },
        }

        ast = AST.from_dict(data)

        assert ast.root == "n1"
        assert len(ast.nodes) == 2
        assert isinstance(ast.nodes["n1"], Number)
        assert ast.nodes["n1"].value == 42

    def test_from_dict_with_refs(self) -> None:
        """Test deserializing AST with references."""

        class Leaf(Node[str], tag="leaf_ast_from_dict_ref"):
            text: str

        class Branch(Node[str], tag="branch_ast_from_dict_ref"):
            left: Ref[Node[str]]
            right: Ref[Node[str]]

        data = {
            "root": "root",
            "nodes": {
                "a": {"tag": "leaf_ast_from_dict_ref", "text": "hello"},
                "b": {"tag": "leaf_ast_from_dict_ref", "text": "world"},
                "root": {
                    "tag": "branch_ast_from_dict_ref",
                    "left": {"tag": "ref", "id": "a"},
                    "right": {"tag": "ref", "id": "b"},
                },
            },
        }

        ast = AST.from_dict(data)

        assert ast.root == "root"
        root_node = ast.nodes["root"]
        assert isinstance(root_node, Branch)
        assert root_node.left.id == "a"
        assert root_node.right.id == "b"

    def test_from_json_simple(self) -> None:
        """Test deserializing AST from JSON string."""

        class Value(Node[int], tag="value_ast_from_json"):
            num: int

        json_str = """{
            "root": "v",
            "nodes": {
                "v": {"tag": "value_ast_from_json", "num": 42}
            }
        }"""

        ast = AST.from_json(json_str)

        assert ast.root == "v"
        assert isinstance(ast.nodes["v"], Value)
        assert ast.nodes["v"].num == 42

    def test_from_json_with_complex_structure(self) -> None:
        """Test deserializing complex AST from JSON."""

        class Const(Node[float], tag="const_ast_from_json_complex"):
            value: float

        class Expr(Node[float], tag="expr_ast_from_json_complex"):
            left: Ref[Node[float]]
            right: Ref[Node[float]]

        json_str = """{
            "root": "result",
            "nodes": {
                "a": {"tag": "const_ast_from_json_complex", "value": 1.5},
                "b": {"tag": "const_ast_from_json_complex", "value": 2.5},
                "result": {
                    "tag": "expr_ast_from_json_complex",
                    "left": {"tag": "ref", "id": "a"},
                    "right": {"tag": "ref", "id": "b"}
                }
            }
        }"""

        ast = AST.from_json(json_str)

        assert ast.root == "result"
        assert len(ast.nodes) == 3
        result_node = ast.nodes["result"]
        assert isinstance(result_node, Expr)


class TestASTRoundTrip:
    """Test round-trip serialization of AST."""

    def test_simple_ast_round_trip(self) -> None:
        """Test simple AST round-trip through dict."""

        class Data(Node[str], tag="data_ast_rt"):
            text: str
            count: int

        original = AST(
            root="d1", nodes={"d1": Data(text="hello", count=1), "d2": Data(text="world", count=2)}
        )

        serialized = original.to_dict()
        deserialized = AST.from_dict(serialized)

        assert deserialized.root == original.root
        assert len(deserialized.nodes) == len(original.nodes)
        assert deserialized.nodes["d1"] == original.nodes["d1"]
        assert deserialized.nodes["d2"] == original.nodes["d2"]

    def test_ast_with_refs_round_trip(self) -> None:
        """Test AST with references round-trip."""

        class Point(Node[tuple[int, int]], tag="point_ast_rt"):
            x: int
            y: int

        class Line(Node[str], tag="line_ast_rt"):
            start: Ref[Node[tuple[int, int]]]
            end: Ref[Node[tuple[int, int]]]

        original = AST(
            root="line",
            nodes={
                "p1": Point(x=0, y=0),
                "p2": Point(x=10, y=10),
                "line": Line(start=Ref(id="p1"), end=Ref(id="p2")),
            },
        )

        serialized = original.to_dict()
        deserialized = AST.from_dict(serialized)

        assert deserialized.root == original.root
        assert len(deserialized.nodes) == 3
        assert deserialized.nodes["p1"] == original.nodes["p1"]
        assert deserialized.nodes["line"] == original.nodes["line"]

    def test_ast_json_round_trip(self) -> None:
        """Test AST round-trip through JSON."""

        class Num(Node[int], tag="num_ast_json_rt"):
            value: int

        class Sum(Node[int], tag="sum_ast_json_rt"):
            left: Ref[Node[int]]
            right: Ref[Node[int]]

        original = AST(
            root="result",
            nodes={
                "a": Num(value=5),
                "b": Num(value=10),
                "result": Sum(left=Ref(id="a"), right=Ref(id="b")),
            },
        )

        json_str = original.to_json()
        deserialized = AST.from_json(json_str)

        assert deserialized.root == original.root
        assert len(deserialized.nodes) == len(original.nodes)
        assert deserialized.nodes["a"] == original.nodes["a"]
        assert deserialized.nodes["b"] == original.nodes["b"]
        assert deserialized.nodes["result"] == original.nodes["result"]


class TestASTWithSharedNodes:
    """Test AST with shared node patterns."""

    def test_diamond_pattern(self) -> None:
        """Test AST with diamond pattern (node reused by multiple parents)."""

        class Val(Node[int], tag="val_diamond"):
            value: int

        class Op(Node[int], tag="op_diamond"):
            left: Ref[Node[int]]
            right: Ref[Node[int]]

        # Diamond: top -> (left, right) -> (both reference bottom)
        ast = AST(
            root="top",
            nodes={
                "bottom": Val(value=1),
                "left": Op(left=Ref(id="bottom"), right=Ref(id="bottom")),
                "right": Op(left=Ref(id="bottom"), right=Ref(id="bottom")),
                "top": Op(left=Ref(id="left"), right=Ref(id="right")),
            },
        )

        # All references to "bottom" should resolve to the same node
        bottom = ast.nodes["bottom"]
        left_node = ast.nodes["left"]
        right_node = ast.nodes["right"]

        assert ast.resolve(left_node.left) is bottom
        assert ast.resolve(left_node.right) is bottom
        assert ast.resolve(right_node.left) is bottom
        assert ast.resolve(right_node.right) is bottom

    def test_shared_subexpression(self) -> None:
        """Test AST with shared subexpression."""

        class Const(Node[float], tag="const_shared_sub"):
            value: float

        class Add(Node[float], tag="add_shared_sub"):
            left: Ref[Node[float]]
            right: Ref[Node[float]]

        class Mul(Node[float], tag="mul_shared_sub"):
            left: Ref[Node[float]]
            right: Ref[Node[float]]

        # Expression: (x + y) * (x + y) where (x + y) is shared
        ast = AST(
            root="result",
            nodes={
                "x": Const(value=2.0),
                "y": Const(value=3.0),
                "sum": Add(left=Ref(id="x"), right=Ref(id="y")),
                "result": Mul(left=Ref(id="sum"), right=Ref(id="sum")),
            },
        )

        result = ast.nodes["result"]
        # Both sides of multiply reference the same sum node
        assert result.left.id == "sum"
        assert result.right.id == "sum"
        assert result.left == result.right


class TestASTEdgeCases:
    """Test edge cases and special scenarios."""

    def test_ast_with_single_node(self) -> None:
        """Test AST with only one node."""

        class Single(Node[int], tag="single_ast"):
            value: int

        ast = AST(root="only", nodes={"only": Single(value=42)})

        assert ast.root == "only"
        assert len(ast.nodes) == 1
        assert ast.nodes["only"].value == 42

    def test_ast_root_can_be_any_node(self) -> None:
        """Test that root can point to any node in the AST."""

        class Node1(Node[int], tag="node1_ast_root"):
            value: int

        class Node2(Node[int], tag="node2_ast_root"):
            value: int

        nodes = {"a": Node1(value=1), "b": Node2(value=2), "c": Node1(value=3)}

        # Root can be any of them
        ast1 = AST(root="a", nodes=nodes)
        assert ast1.root == "a"

        ast2 = AST(root="b", nodes=nodes)
        assert ast2.root == "b"

        ast3 = AST(root="c", nodes=nodes)
        assert ast3.root == "c"

    def test_ast_with_node_containing_none(self) -> None:
        """Test AST with node that has None field."""

        class Optional(Node[str], tag="optional_ast_edge"):
            required: str
            optional: int | None

        ast = AST(root="opt", nodes={"opt": Optional(required="value", optional=None)})

        result = ast.to_dict()
        assert result["nodes"]["opt"]["optional"] is None

        # Round trip
        restored = AST.from_dict(result)
        assert restored.nodes["opt"].optional is None

    def test_empty_ast_serialization(self) -> None:
        """Test serializing empty AST."""
        ast = AST(root="", nodes={})

        result = ast.to_dict()
        assert result["root"] == ""
        assert result["nodes"] == {}

        # Round trip
        restored = AST.from_dict(result)
        assert restored.root == ""
        assert len(restored.nodes) == 0


class TestASTIntegrationExamples:
    """Test complete real-world-like examples using AST."""

    def test_expression_tree_example(self) -> None:
        """Test complete expression tree: (a + b) * c."""

        class Var(Node[float], tag="var_example"):
            name: str

        class BinOp(Node[float], tag="binop_example"):
            op: str
            left: Ref[Node[float]]
            right: Ref[Node[float]]

        ast = AST(
            root="result",
            nodes={
                "a": Var(name="a"),
                "b": Var(name="b"),
                "c": Var(name="c"),
                "sum": BinOp(op="+", left=Ref(id="a"), right=Ref(id="b")),
                "result": BinOp(op="*", left=Ref(id="sum"), right=Ref(id="c")),
            },
        )

        # Verify structure
        assert ast.root == "result"
        result_node = ast.nodes["result"]
        assert result_node.op == "*"
        assert result_node.left.id == "sum"
        assert result_node.right.id == "c"

        # Verify serialization
        json_str = ast.to_json()
        restored = AST.from_json(json_str)
        assert restored == ast

    def test_dataflow_graph_example(self) -> None:
        """Test dataflow graph with shared inputs."""

        class Input(Node[int], tag="input_dataflow"):
            source: str

        class Transform(Node[int], tag="transform_dataflow"):
            func: str
            input: Ref[Node[int]]

        class Merge(Node[int], tag="merge_dataflow"):
            left: Ref[Node[int]]
            right: Ref[Node[int]]

        # Graph: input -> (transform1, transform2) -> merge
        ast = AST(
            root="output",
            nodes={
                "input": Input(source="data.csv"),
                "t1": Transform(func="normalize", input=Ref(id="input")),
                "t2": Transform(func="scale", input=Ref(id="input")),
                "output": Merge(left=Ref(id="t1"), right=Ref(id="t2")),
            },
        )

        # Both transforms reference the same input
        t1 = ast.nodes["t1"]
        t2 = ast.nodes["t2"]
        assert ast.resolve(t1.input) is ast.nodes["input"]
        assert ast.resolve(t2.input) is ast.nodes["input"]

        # Verify round-trip
        serialized = ast.to_dict()
        restored = AST.from_dict(serialized)
        assert restored.root == ast.root
        assert len(restored.nodes) == len(ast.nodes)


class TestInterpreterBasics:
    """Test basic Interpreter functionality."""

    def test_interpreter_is_abstract(self) -> None:
        """Test that Interpreter cannot be instantiated directly."""

        class Num(Node[int], tag="num_interp_abstract"):
            value: int

        ast = AST(root="n", nodes={"n": Num(value=1)})

        with pytest.raises(TypeError, match="abstract"):
            Interpreter(ast, None)  # type: ignore[abstract]

    def test_simple_interpreter(self) -> None:
        """Test a simple interpreter that evaluates constants."""

        class Const(Node[float], tag="const_interp_simple"):
            value: float

        class Calculator(Interpreter[None, float]):
            def eval(self, node: Node[Any]) -> float:
                match node:
                    case Const(value=v):
                        return v
                    case _:
                        raise NotImplementedError(f"Unknown node: {type(node)}")

        ast = AST(root="c", nodes={"c": Const(value=42.0)})
        result = Calculator(ast, None).run()

        assert result == 42.0

    def test_interpreter_with_context(self) -> None:
        """Test interpreter that uses context for variable lookup."""

        class Var(Node[float], tag="var_interp_ctx"):
            name: str

        class Calculator(Interpreter[dict[str, float], float]):
            def eval(self, node: Node[Any]) -> float:
                match node:
                    case Var(name=n):
                        return self.ctx[n]
                    case _:
                        raise NotImplementedError(f"Unknown node: {type(node)}")

        ast = AST(root="x", nodes={"x": Var(name="x")})
        result = Calculator(ast, {"x": 10.0, "y": 20.0}).run()

        assert result == 10.0

    def test_interpreter_has_ast_access(self) -> None:
        """Test that interpreter has access to the AST."""

        class Num(Node[int], tag="num_interp_ast_access"):
            value: int

        class Inspector(Interpreter[None, int]):
            def eval(self, node: Node[Any]) -> int:
                # Access ast from within eval
                return len(self.ast.nodes)

        ast = AST(root="a", nodes={"a": Num(value=1), "b": Num(value=2), "c": Num(value=3)})
        result = Inspector(ast, None).run()

        assert result == 3


class TestInterpreterResolve:
    """Test Interpreter.resolve() functionality."""

    def test_resolve_returns_node(self) -> None:
        """Test that resolve returns the referenced node."""

        class Val(Node[int], tag="val_interp_resolve"):
            value: int

        class Wrapper(Node[int], tag="wrapper_interp_resolve"):
            inner: Ref[Node[int]]

        class Evaluator(Interpreter[None, int]):
            def eval(self, node: Node[Any]) -> int:
                match node:
                    case Val(value=v):
                        return v
                    case Wrapper(inner=ref):
                        resolved = self.resolve(ref)
                        return self.eval(resolved)
                    case _:
                        raise NotImplementedError

        ast = AST(
            root="w",
            nodes={
                "v": Val(value=42),
                "w": Wrapper(inner=Ref(id="v")),
            },
        )
        result = Evaluator(ast, None).run()

        assert result == 42

    def test_resolve_with_refs_in_expression(self) -> None:
        """Test resolving refs in a binary expression."""

        class Const(Node[float], tag="const_interp_expr"):
            value: float

        class Add(Node[float], tag="add_interp_expr"):
            left: Ref[Node[float]]
            right: Ref[Node[float]]

        class Calculator(Interpreter[None, float]):
            def eval(self, node: Node[Any]) -> float:
                match node:
                    case Const(value=v):
                        return v
                    case Add(left=l, right=r):
                        return self.eval(self.resolve(l)) + self.eval(self.resolve(r))
                    case _:
                        raise NotImplementedError

        ast = AST(
            root="sum",
            nodes={
                "a": Const(value=10.0),
                "b": Const(value=32.0),
                "sum": Add(left=Ref(id="a"), right=Ref(id="b")),
            },
        )
        result = Calculator(ast, None).run()

        assert result == 42.0


class TestInterpreterWithSharedNodes:
    """Test Interpreter with DAG patterns (shared nodes)."""

    def test_shared_node_evaluated_multiple_times(self) -> None:
        """Test that shared nodes are evaluated each time (no built-in memoization)."""

        class Counter(Node[int], tag="counter_interp_shared"):
            value: int

        class Add(Node[int], tag="add_interp_shared"):
            left: Ref[Node[int]]
            right: Ref[Node[int]]

        class CountingEvaluator(Interpreter[None, int]):
            def __init__(self, ast: AST, ctx: None) -> None:
                super().__init__(ast, ctx)
                self.eval_count = 0

            def eval(self, node: Node[Any]) -> int:
                self.eval_count += 1
                match node:
                    case Counter(value=v):
                        return v
                    case Add(left=l, right=r):
                        return self.eval(self.resolve(l)) + self.eval(self.resolve(r))
                    case _:
                        raise NotImplementedError

        # x is shared: result = x + x
        ast = AST(
            root="result",
            nodes={
                "x": Counter(value=5),
                "result": Add(left=Ref(id="x"), right=Ref(id="x")),
            },
        )

        evaluator = CountingEvaluator(ast, None)
        result = evaluator.run()

        assert result == 10
        # x is evaluated twice (once for left, once for right), plus result itself
        assert evaluator.eval_count == 3

    def test_diamond_pattern_evaluation(self) -> None:
        """Test evaluation of diamond pattern."""

        class Const(Node[float], tag="const_interp_diamond"):
            value: float

        class Mul(Node[float], tag="mul_interp_diamond"):
            left: Ref[Node[float]]
            right: Ref[Node[float]]

        class Add(Node[float], tag="add_interp_diamond"):
            left: Ref[Node[float]]
            right: Ref[Node[float]]

        class Calculator(Interpreter[None, float]):
            def eval(self, node: Node[Any]) -> float:
                match node:
                    case Const(value=v):
                        return v
                    case Mul(left=l, right=r):
                        return self.eval(self.resolve(l)) * self.eval(self.resolve(r))
                    case Add(left=l, right=r):
                        return self.eval(self.resolve(l)) + self.eval(self.resolve(r))
                    case _:
                        raise NotImplementedError

        # Expression: (x * y) + (x * y) where (x * y) is shared
        ast = AST(
            root="result",
            nodes={
                "x": Const(value=3.0),
                "y": Const(value=4.0),
                "product": Mul(left=Ref(id="x"), right=Ref(id="y")),
                "result": Add(left=Ref(id="product"), right=Ref(id="product")),
            },
        )

        result = Calculator(ast, None).run()
        assert result == 24.0  # (3 * 4) + (3 * 4) = 12 + 12 = 24


class TestInterpreterUserMemoization:
    """Test that users can implement their own memoization."""

    def test_user_implemented_memoization(self) -> None:
        """Test user-implemented memoization pattern."""

        class Const(Node[int], tag="const_interp_memo"):
            value: int

        class Add(Node[int], tag="add_interp_memo"):
            left: Ref[Node[int]]
            right: Ref[Node[int]]

        class MemoizingCalculator(Interpreter[None, int]):
            def __init__(self, ast: AST, ctx: None) -> None:
                super().__init__(ast, ctx)
                self._cache: dict[str, int] = {}
                self.eval_count = 0

            def eval_ref(self, ref: Ref[Node[int]]) -> int:
                """Evaluate a ref with memoization."""
                if ref.id not in self._cache:
                    self._cache[ref.id] = self.eval(self.resolve(ref))
                return self._cache[ref.id]

            def eval(self, node: Node[Any]) -> int:
                self.eval_count += 1
                match node:
                    case Const(value=v):
                        return v
                    case Add(left=l, right=r):
                        return self.eval_ref(l) + self.eval_ref(r)
                    case _:
                        raise NotImplementedError

        # x is shared: result = x + x
        ast = AST(
            root="result",
            nodes={
                "x": Const(value=5),
                "result": Add(left=Ref(id="x"), right=Ref(id="x")),
            },
        )

        evaluator = MemoizingCalculator(ast, None)
        result = evaluator.run()

        assert result == 10
        # With memoization, x is only evaluated once
        assert evaluator.eval_count == 2  # result + x (x cached for second use)


class TestInterpreterComplexExamples:
    """Test complete real-world-like interpreter examples."""

    def test_arithmetic_expression_evaluator(self) -> None:
        """Test complete arithmetic expression evaluator."""

        class Const(Node[float], tag="const_interp_arith"):
            value: float

        class Var(Node[float], tag="var_interp_arith"):
            name: str

        class BinOp(Node[float], tag="binop_interp_arith"):
            op: str
            left: Ref[Node[float]]
            right: Ref[Node[float]]

        class ArithmeticEvaluator(Interpreter[dict[str, float], float]):
            def eval(self, node: Node[Any]) -> float:
                match node:
                    case Const(value=v):
                        return v
                    case Var(name=n):
                        return self.ctx[n]
                    case BinOp(op="+", left=l, right=r):
                        return self.eval(self.resolve(l)) + self.eval(self.resolve(r))
                    case BinOp(op="-", left=l, right=r):
                        return self.eval(self.resolve(l)) - self.eval(self.resolve(r))
                    case BinOp(op="*", left=l, right=r):
                        return self.eval(self.resolve(l)) * self.eval(self.resolve(r))
                    case BinOp(op="/", left=l, right=r):
                        return self.eval(self.resolve(l)) / self.eval(self.resolve(r))
                    case _:
                        raise NotImplementedError(f"Unknown node: {type(node)}")

        # Expression: (x + 2) * (y - 1)
        ast = AST(
            root="result",
            nodes={
                "x": Var(name="x"),
                "y": Var(name="y"),
                "two": Const(value=2.0),
                "one": Const(value=1.0),
                "sum": BinOp(op="+", left=Ref(id="x"), right=Ref(id="two")),
                "diff": BinOp(op="-", left=Ref(id="y"), right=Ref(id="one")),
                "result": BinOp(op="*", left=Ref(id="sum"), right=Ref(id="diff")),
            },
        )

        result = ArithmeticEvaluator(ast, {"x": 3.0, "y": 5.0}).run()
        # (3 + 2) * (5 - 1) = 5 * 4 = 20
        assert result == 20.0

    def test_string_concatenation_interpreter(self) -> None:
        """Test interpreter for string operations."""

        class StrLiteral(Node[str], tag="strlit_interp"):
            value: str

        class Concat(Node[str], tag="concat_interp"):
            left: Ref[Node[str]]
            right: Ref[Node[str]]

        class StringInterpreter(Interpreter[None, str]):
            def eval(self, node: Node[Any]) -> str:
                match node:
                    case StrLiteral(value=v):
                        return v
                    case Concat(left=l, right=r):
                        return self.eval(self.resolve(l)) + self.eval(self.resolve(r))
                    case _:
                        raise NotImplementedError

        ast = AST(
            root="result",
            nodes={
                "hello": StrLiteral(value="Hello"),
                "space": StrLiteral(value=" "),
                "world": StrLiteral(value="World"),
                "hello_space": Concat(left=Ref(id="hello"), right=Ref(id="space")),
                "result": Concat(left=Ref(id="hello_space"), right=Ref(id="world")),
            },
        )

        result = StringInterpreter(ast, None).run()
        assert result == "Hello World"

    def test_interpreter_with_inline_and_ref_nodes(self) -> None:
        """Test interpreter that handles both inline nodes and refs."""

        class Const(Node[int], tag="const_interp_mixed"):
            value: int

        class Add(Node[int], tag="add_interp_mixed"):
            left: Node[int] | Ref[Node[int]]
            right: Node[int] | Ref[Node[int]]

        class MixedEvaluator(Interpreter[None, int]):
            def eval(self, node: Node[Any]) -> int:
                match node:
                    case Const(value=v):
                        return v
                    case Add(left=l, right=r):
                        # Handle both inline nodes and refs
                        left_val = (
                            self.eval(self.resolve(l)) if isinstance(l, Ref) else self.eval(l)
                        )
                        right_val = (
                            self.eval(self.resolve(r)) if isinstance(r, Ref) else self.eval(r)
                        )
                        return left_val + right_val
                    case _:
                        raise NotImplementedError

        # Mix of inline and ref nodes
        ast = AST(
            root="result",
            nodes={
                "shared": Const(value=10),
                # result has inline left and ref right
                "result": Add(left=Const(value=5), right=Ref(id="shared")),
            },
        )

        result = MixedEvaluator(ast, None).run()
        assert result == 15
