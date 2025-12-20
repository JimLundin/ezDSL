"""Integration tests for typeDSL - end-to-end workflows."""

import pytest

from typedsl import (
    AST,
    Child,
    Node,
    NodeRef,
    Ref,
    all_schemas,
    from_json,
    node_schema,
    to_json,
)
from typedsl.types import (
    FloatType,
    IntType,
    ListType,
    NodeType,
    RefType,
    StrType,
)


class TestCompleteExpressionTreeWorkflow:
    """Test complete workflow: define nodes, build tree, serialize, deserialize."""

    def test_simple_expression_tree(self) -> None:
        """Test: Define -> Build -> Serialize -> Deserialize -> Verify."""

        # Step 1: Define node types
        class Literal(Node[float], tag="lit_integration"):
            value: float

        class BinOp(Node[float], tag="binop_integration"):
            op: str
            left: Node[float]
            right: Node[float]

        # Step 2: Build expression tree: (1.5 + 2.5) * 3.0
        tree = BinOp(
            op="*",
            left=BinOp(op="+", left=Literal(value=1.5), right=Literal(value=2.5)),
            right=Literal(value=3.0),
        )

        # Step 3: Serialize to JSON
        json_str = to_json(tree)

        # Step 4: Deserialize from JSON
        restored = from_json(json_str)

        # Step 5: Verify structure preserved
        assert restored == tree
        assert isinstance(restored, BinOp)
        assert restored.op == "*"
        assert isinstance(restored.left, BinOp)
        assert restored.left.op == "+"

    def test_expression_with_schema_extraction(self) -> None:
        """Test workflow with schema extraction."""

        # Define nodes
        class Const(Node[int], tag="const_schema_int"):
            value: int

        class Add(Node[int], tag="add_schema_int"):
            left: Node[int]
            right: Node[int]

        # Extract schemas
        const_schema = node_schema(Const)
        add_schema = node_schema(Add)

        # Verify schemas
        assert const_schema.tag == "const_schema_int"
        assert len(const_schema.fields) == 1
        assert const_schema.fields[0].name == "value"

        assert add_schema.tag == "add_schema_int"
        assert len(add_schema.fields) == 2

        # Build and serialize
        expr = Add(left=Const(value=1), right=Const(value=2))
        json_str = to_json(expr)

        # Deserialize and verify
        restored = from_json(json_str)
        assert restored == expr


class TestASTWithReferencesWorkflow:
    """Test complete workflow using AST with references."""

    def test_build_ast_with_shared_nodes(self) -> None:
        """Test building AST with shared nodes via references."""

        # Step 1: Define nodes that use references
        class Value(Node[float], tag="value_ast_workflow"):
            num: float

        class Compute(Node[float], tag="compute_ast_workflow"):
            left: Ref[Node[float]]
            right: Ref[Node[float]]

        # Step 2: Build AST with shared node
        # Expression: x + x (where x is shared)
        ast = AST(
            root="result",
            nodes={
                "x": Value(num=5.0),
                "result": Compute(left=Ref(id="x"), right=Ref(id="x")),
            },
        )

        # Step 3: Verify reference resolution
        x_node = ast.nodes["x"]
        result_node = ast.nodes["result"]

        assert isinstance(result_node, Compute)
        resolved_left = ast.resolve(result_node.left)
        resolved_right = ast.resolve(result_node.right)

        # Both should be the same object
        assert resolved_left is x_node
        assert resolved_right is x_node

        # Step 4: Serialize AST
        json_str = ast.to_json()

        # Step 5: Deserialize AST
        restored_ast = AST.from_json(json_str)

        # Step 6: Verify structure preserved
        assert restored_ast.root == ast.root
        assert len(restored_ast.nodes) == len(ast.nodes)
        assert restored_ast.nodes["x"] == ast.nodes["x"]
        assert restored_ast.nodes["result"] == ast.nodes["result"]

    def test_complex_dataflow_graph(self) -> None:
        """Test complex dataflow with multiple shared nodes."""

        # Define nodes
        class Input(Node[str], tag="input_dataflow_int"):
            source: str

        class Transform(Node[str], tag="transform_dataflow_int"):
            func: str
            input: Ref[Node[str]]

        class Join(Node[str], tag="join_dataflow_int"):
            left: Ref[Node[str]]
            right: Ref[Node[str]]

        # Build complex graph:
        # input -> (t1, t2, t3) -> (j1, j2) -> final
        ast = AST(
            root="final",
            nodes={
                "input": Input(source="data.txt"),
                "t1": Transform(func="upper", input=Ref(id="input")),
                "t2": Transform(func="lower", input=Ref(id="input")),
                "t3": Transform(func="strip", input=Ref(id="input")),
                "j1": Join(left=Ref(id="t1"), right=Ref(id="t2")),
                "j2": Join(left=Ref(id="j1"), right=Ref(id="t3")),
                "final": Transform(func="finalize", input=Ref(id="j2")),
            },
        )

        # Verify graph structure
        assert len(ast.nodes) == 7

        # Serialize and deserialize
        json_str = ast.to_json()
        restored = AST.from_json(json_str)

        # Verify restoration
        assert restored.root == "final"
        assert len(restored.nodes) == 7

        # Verify relationships preserved
        final_node = restored.nodes["final"]
        j2_node = restored.nodes["j2"]

        assert isinstance(final_node, Transform)
        assert final_node.input.id == "j2"

        assert isinstance(j2_node, Join)
        assert j2_node.left.id == "j1"


class TestGenericNodeWorkflow:
    """Test workflow with generic nodes."""

    def test_generic_node_instantiation_and_serialization(self) -> None:
        """Test creating and serializing generic nodes."""

        # Define generic node
        class Container[T](Node[list[T]], tag="container_generic_int"):
            items: list[T]

        # Create instances with different types
        int_container = Container(items=[1, 2, 3])
        str_container = Container(items=["a", "b", "c"])

        # Serialize both
        int_json = to_json(int_container)
        str_json = to_json(str_container)

        # Deserialize
        restored_int = from_json(int_json)
        restored_str = from_json(str_json)

        # Verify
        assert restored_int == int_container
        assert restored_str == str_container

        assert isinstance(restored_int, Container)
        assert restored_int.items == [1, 2, 3]

        assert isinstance(restored_str, Container)
        assert restored_str.items == ["a", "b", "c"]

    def test_generic_node_schema_extraction(self) -> None:
        """Test extracting schema from generic node."""

        # Define generic node
        class Wrapper[T](Node[T], tag="wrapper_generic_int"):
            value: T
            label: str

        # Extract schema
        schema = node_schema(Wrapper)

        # Verify schema structure
        assert schema.tag == "wrapper_generic_int"
        assert len(schema.type_params) == 1
        assert schema.type_params[0].name == "T"
        assert len(schema.fields) == 2

        # Verify field types
        field_names = [f.name for f in schema.fields]
        assert "value" in field_names
        assert "label" in field_names


class TestSchemaRegistryWorkflow:
    """Test working with schema registry."""

    def test_register_and_query_schemas(self) -> None:
        """Test registering nodes and querying their schemas."""

        # Define several nodes
        class NodeA(Node[int], tag="node_a_registry"):
            x: int

        class NodeB(Node[str], tag="node_b_registry"):
            text: str

        class NodeC(Node[float], tag="node_c_registry"):
            value: float

        # Get all schemas
        schemas = all_schemas()

        # Verify our nodes are registered
        assert "node_a_registry" in schemas
        assert "node_b_registry" in schemas
        assert "node_c_registry" in schemas

        # Verify schema details
        schema_a = schemas["node_a_registry"]
        assert schema_a.tag == "node_a_registry"
        assert len(schema_a.fields) == 1


class TestTypeExtractionWorkflow:
    """Test type extraction and introspection."""

    def test_extract_node_field_types(self) -> None:
        """Test extracting field types from node definition."""

        # Define node with various field types
        class ComplexNode(Node[str], tag="complex_type_int"):
            int_field: int
            str_field: str
            list_field: list[int]
            node_field: Node[float]
            ref_field: Ref[Node[int]]

        # Extract schema
        schema = node_schema(ComplexNode)

        # Extract field types
        fields_by_name = {f.name: f.type for f in schema.fields}

        # Verify type extraction
        assert isinstance(fields_by_name["int_field"], IntType)
        assert isinstance(fields_by_name["str_field"], StrType)
        assert isinstance(fields_by_name["list_field"], ListType)
        assert isinstance(fields_by_name["list_field"].element, IntType)
        assert isinstance(fields_by_name["node_field"], NodeType)
        assert isinstance(fields_by_name["node_field"].returns, FloatType)
        assert isinstance(fields_by_name["ref_field"], RefType)


class TestEndToEndUserScenarios:
    """Test realistic end-to-end user scenarios."""

    def test_calculator_dsl(self) -> None:
        """Test building a simple calculator DSL."""

        # Define calculator nodes
        class Number(Node[float], tag="number_calc"):
            value: float

        class Add(Node[float], tag="add_calc"):
            left: Node[float]
            right: Node[float]

        class Multiply(Node[float], tag="multiply_calc"):
            left: Node[float]
            right: Node[float]

        class Negate(Node[float], tag="negate_calc"):
            operand: Node[float]

        # Build expression: -(2 + 3) * 4
        expr = Multiply(
            left=Negate(operand=Add(left=Number(value=2), right=Number(value=3))),
            right=Number(value=4),
        )

        # Serialize
        json_str = to_json(expr)

        # Deserialize
        restored = from_json(json_str)

        # Verify structure
        assert isinstance(restored, Multiply)
        assert isinstance(restored.left, Negate)
        assert isinstance(restored.left.operand, Add)
        assert restored.right.value == 4

    def test_query_builder_dsl(self) -> None:
        """Test building a query builder DSL with references."""

        # Define query nodes
        class Table(Node[str], tag="table_query"):
            name: str

        class Filter(Node[str], tag="filter_query"):
            source: Ref[Node[str]]
            condition: str

        class Select(Node[str], tag="select_query"):
            source: Ref[Node[str]]
            columns: list[str]

        class Join(Node[str], tag="join_query"):
            left: Ref[Node[str]]
            right: Ref[Node[str]]
            on: str

        # Build query AST
        ast = AST(
            root="result",
            nodes={
                "users": Table(name="users"),
                "orders": Table(name="orders"),
                "active_users": Filter(source=Ref(id="users"), condition="active=true"),
                "join": Join(
                    left=Ref(id="active_users"),
                    right=Ref(id="orders"),
                    on="user_id",
                ),
                "result": Select(
                    source=Ref(id="join"),
                    columns=["name", "order_id", "total"],
                ),
            },
        )

        # Serialize and deserialize
        json_str = ast.to_json()
        restored = AST.from_json(json_str)

        # Verify query structure preserved
        assert restored.root == "result"
        result_node = restored.nodes["result"]
        assert isinstance(result_node, Select)
        assert result_node.columns == ["name", "order_id", "total"]
        assert result_node.source.id == "join"

    def test_ml_pipeline_dsl(self) -> None:
        """Test building a machine learning pipeline DSL."""

        # Define ML pipeline nodes
        class DataSource(Node[str], tag="datasource_ml"):
            path: str

        class Preprocess(Node[str], tag="preprocess_ml"):
            input: Ref[Node[str]]
            steps: list[str]

        class Model(Node[str], tag="model_ml"):
            name: str
            params: dict[str, float]

        class Train(Node[str], tag="train_ml"):
            data: Ref[Node[str]]
            model: Ref[Node[str]]

        class Evaluate(Node[str], tag="evaluate_ml"):
            trained: Ref[Node[str]]
            test_data: Ref[Node[str]]

        # Build pipeline
        ast = AST(
            root="evaluation",
            nodes={
                "raw_data": DataSource(path="data.csv"),
                "preprocessed": Preprocess(
                    input=Ref(id="raw_data"),
                    steps=["normalize", "remove_nulls"],
                ),
                "model": Model(name="random_forest", params={"n_estimators": 100.0}),
                "training": Train(data=Ref(id="preprocessed"), model=Ref(id="model")),
                "test_data": DataSource(path="test.csv"),
                "test_prep": Preprocess(input=Ref(id="test_data"), steps=["normalize"]),
                "evaluation": Evaluate(
                    trained=Ref(id="training"),
                    test_data=Ref(id="test_prep"),
                ),
            },
        )

        # Round-trip through JSON
        json_str = ast.to_json()
        restored = AST.from_json(json_str)

        # Verify pipeline intact
        assert len(restored.nodes) == 7
        eval_node = restored.nodes["evaluation"]
        assert eval_node.trained.id == "training"
        assert eval_node.test_data.id == "test_prep"


class TestTypeAliasesIntegration:
    """Test integration with type aliases."""

    def test_using_child_type_alias(self) -> None:
        """Test using Child[T] type alias in practice."""

        # Define node using Child type alias
        class FlexNode(Node[int], tag="flex_child"):
            # Can accept either inline node or reference
            input: Child[int]

        # Test with inline node
        class Value(Node[int], tag="value_child"):
            num: int

        inline_node = FlexNode(input=Value(num=42))
        assert isinstance(inline_node.input, Node)

        # Test with reference (in AST context)
        ref_node = FlexNode(input=Ref[Node[int]](id="some_value"))
        assert isinstance(ref_node.input, Ref)

        # Both serialize correctly
        inline_json = to_json(inline_node)
        ref_json = to_json(ref_node)

        restored_inline = from_json(inline_json)
        restored_ref = from_json(ref_json)

        assert restored_inline == inline_node
        assert restored_ref == ref_node

    def test_using_node_ref_type_alias(self) -> None:
        """Test using NodeRef[T] type alias."""

        # NodeRef[T] is equivalent to Ref[Node[T]]
        class Container(Node[str], tag="container_noderef"):
            target: NodeRef[str]

        node = Container(target=Ref[Node[str]](id="target-123"))
        assert node.target.id == "target-123"

        # Round-trip
        json_str = to_json(node)
        restored = from_json(json_str)
        assert restored == node


class TestErrorRecovery:
    """Test graceful error handling in integration scenarios."""

    def test_deserialize_with_missing_node_definition(self) -> None:
        """Test attempting to deserialize unknown node type."""
        # JSON with a tag that doesn't exist
        json_str = '{"tag": "unknown_node_type_xyz", "value": 42}'

        with pytest.raises(ValueError, match="Unknown tag"):
            from_json(json_str)

    def test_resolve_nonexistent_reference(self) -> None:
        """Test resolving reference that doesn't exist in AST."""

        class Dummy(Node[int], tag="dummy_err"):
            value: int

        ast = AST(root="node1", nodes={"node1": Dummy(value=1)})

        # Try to resolve non-existent reference
        ref = Ref[Node[int]](id="nonexistent")

        with pytest.raises(KeyError):
            ast.resolve(ref)
