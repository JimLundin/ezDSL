"""Tests for typedsl.adapters module."""

import pytest

from typedsl.adapters import JSONAdapter
from typedsl.nodes import Node, Ref
from typedsl.schema import node_schema
from typedsl.types import (
    BoolType,
    DictType,
    FloatType,
    IntType,
    ListType,
    NodeType,
    RefType,
    StrType,
    UnionType,
)


class TestJSONAdapterSerializeNode:
    """Test JSONAdapter.serialize_node() method."""

    def test_serialize_simple_node(self) -> None:
        """Test serializing a simple node."""

        class Value(Node[int], tag="value_adapter"):
            num: int

        adapter = JSONAdapter()
        node = Value(num=42)
        result = adapter.serialize_node(node)

        assert result == {"tag": "value_adapter", "num": 42}

    def test_serialize_node_with_multiple_fields(self) -> None:
        """Test serializing node with multiple fields."""

        class Record(Node[str], tag="record_adapter"):
            name: str
            age: int
            active: bool

        adapter = JSONAdapter()
        node = Record(name="Bob", age=25, active=True)
        result = adapter.serialize_node(node)

        assert result == {
            "tag": "record_adapter",
            "name": "Bob",
            "age": 25,
            "active": True,
        }

    def test_serialize_nested_nodes(self) -> None:
        """Test serializing nested nodes."""

        class Leaf(Node[int], tag="leaf_adapter"):
            value: int

        class Branch(Node[int], tag="branch_adapter"):
            left: Node[int]
            right: Node[int]

        adapter = JSONAdapter()
        tree = Branch(left=Leaf(value=1), right=Leaf(value=2))
        result = adapter.serialize_node(tree)

        assert result == {
            "tag": "branch_adapter",
            "left": {"tag": "leaf_adapter", "value": 1},
            "right": {"tag": "leaf_adapter", "value": 2},
        }

    def test_serialize_node_with_ref(self) -> None:
        """Test serializing node containing a Ref."""

        class RefNode(Node[int], tag="ref_node_adapter"):
            target: Ref[Node[int]]

        adapter = JSONAdapter()
        node = RefNode(target=Ref(id="node-123"))
        result = adapter.serialize_node(node)

        assert result == {
            "tag": "ref_node_adapter",
            "target": {"tag": "ref", "id": "node-123"},
        }

    def test_serialize_node_with_list(self) -> None:
        """Test serializing node with list field."""

        class ListNode(Node[list[int]], tag="list_node_adapter"):
            items: list[int]

        adapter = JSONAdapter()
        node = ListNode(items=[1, 2, 3])
        result = adapter.serialize_node(node)

        assert result == {"tag": "list_node_adapter", "items": [1, 2, 3]}

    def test_serialize_node_with_dict(self) -> None:
        """Test serializing node with dict field."""

        class DictNode(Node[dict[str, int]], tag="dict_node_adapter"):
            data: dict[str, int]

        adapter = JSONAdapter()
        node = DictNode(data={"a": 1, "b": 2})
        result = adapter.serialize_node(node)

        assert result == {"tag": "dict_node_adapter", "data": {"a": 1, "b": 2}}

    def test_serialize_node_with_nested_list(self) -> None:
        """Test serializing node with nested list structure."""

        class NestedListNode(Node[list[list[int]]], tag="nested_list_adapter"):
            matrix: list[list[int]]

        adapter = JSONAdapter()
        node = NestedListNode(matrix=[[1, 2], [3, 4]])
        result = adapter.serialize_node(node)

        assert result == {"tag": "nested_list_adapter", "matrix": [[1, 2], [3, 4]]}

    def test_serialize_excludes_private_fields(self) -> None:
        """Test that fields starting with _ are excluded from serialization."""

        class NodeWithPrivate(Node[int], tag="private_field_adapter"):
            public_value: int
            _private_value: int = 999

        adapter = JSONAdapter()
        node = NodeWithPrivate(public_value=42)
        result = adapter.serialize_node(node)

        assert result == {"tag": "private_field_adapter", "public_value": 42}
        assert "_private_value" not in result


class TestJSONAdapterDeserializeNode:
    """Test JSONAdapter.deserialize_node() method."""

    def test_deserialize_simple_node(self) -> None:
        """Test deserializing a simple node."""

        class Item(Node[str], tag="item_deser"):
            name: str

        adapter = JSONAdapter()
        data = {"tag": "item_deser", "name": "test"}
        result = adapter.deserialize_node(data)

        assert isinstance(result, Item)
        assert result.name == "test"

    def test_deserialize_node_with_multiple_fields(self) -> None:
        """Test deserializing node with multiple fields."""

        class User(Node[str], tag="user_deser"):
            username: str
            email: str
            score: int

        adapter = JSONAdapter()
        data = {
            "tag": "user_deser",
            "username": "alice",
            "email": "alice@example.com",
            "score": 100,
        }
        result = adapter.deserialize_node(data)

        assert isinstance(result, User)
        assert result.username == "alice"
        assert result.email == "alice@example.com"
        assert result.score == 100

    def test_deserialize_nested_nodes(self) -> None:
        """Test deserializing nested nodes."""

        class Point(Node[tuple[int, int]], tag="point_deser"):
            x: int
            y: int

        class Line(Node[str], tag="line_deser"):
            start: Node[tuple[int, int]]
            end: Node[tuple[int, int]]

        adapter = JSONAdapter()
        data = {
            "tag": "line_deser",
            "start": {"tag": "point_deser", "x": 0, "y": 0},
            "end": {"tag": "point_deser", "x": 10, "y": 10},
        }
        result = adapter.deserialize_node(data)

        assert isinstance(result, Line)
        assert isinstance(result.start, Point)
        assert isinstance(result.end, Point)
        assert result.start.x == 0
        assert result.end.x == 10

    def test_deserialize_node_with_ref(self) -> None:
        """Test deserializing node containing Ref."""

        class NodeWithRef(Node[int], tag="node_with_ref_deser"):
            ref_field: Ref[Node[int]]

        adapter = JSONAdapter()
        data = {
            "tag": "node_with_ref_deser",
            "ref_field": {"tag": "ref", "id": "ref-id-123"},
        }
        result = adapter.deserialize_node(data)

        assert isinstance(result, NodeWithRef)
        assert isinstance(result.ref_field, Ref)
        assert result.ref_field.id == "ref-id-123"

    def test_deserialize_unknown_tag_raises_error(self) -> None:
        """Test that unknown node tag raises ValueError."""
        adapter = JSONAdapter()
        data = {"tag": "unknown_node_tag_xyz"}

        with pytest.raises(ValueError, match="Unknown node tag"):
            adapter.deserialize_node(data)

    def test_deserialize_node_with_missing_fields(self) -> None:
        """Test deserializing node when some fields are missing from data."""

        class Optional(Node[str], tag="optional_deser"):
            required: str
            optional: int | None = None

        adapter = JSONAdapter()
        # Only provide required field
        data = {"tag": "optional_deser", "required": "value"}
        result = adapter.deserialize_node(data)

        assert isinstance(result, Optional)
        assert result.required == "value"
        # optional field gets default value
        assert result.optional is None


class TestJSONAdapterSerializeTypeDef:
    """Test JSONAdapter.serialize_typedef() method."""

    def test_serialize_primitive_typedef(self) -> None:
        """Test serializing primitive TypeDef."""
        adapter = JSONAdapter()

        int_type = IntType()
        assert adapter.serialize_typedef(int_type) == {"tag": "int"}

        float_type = FloatType()
        assert adapter.serialize_typedef(float_type) == {"tag": "float"}

        str_type = StrType()
        assert adapter.serialize_typedef(str_type) == {"tag": "str"}

        bool_type = BoolType()
        assert adapter.serialize_typedef(bool_type) == {"tag": "bool"}

    def test_serialize_list_typedef(self) -> None:
        """Test serializing ListType."""
        adapter = JSONAdapter()
        list_type = ListType(element=IntType())
        result = adapter.serialize_typedef(list_type)

        assert result == {"tag": "list", "element": {"tag": "int"}}

    def test_serialize_dict_typedef(self) -> None:
        """Test serializing DictType."""
        adapter = JSONAdapter()
        dict_type = DictType(key=StrType(), value=FloatType())
        result = adapter.serialize_typedef(dict_type)

        assert result == {
            "tag": "dict",
            "key": {"tag": "str"},
            "value": {"tag": "float"},
        }

    def test_serialize_nested_typedef(self) -> None:
        """Test serializing nested TypeDef."""
        adapter = JSONAdapter()
        # list[list[int]]
        nested = ListType(element=ListType(element=IntType()))
        result = adapter.serialize_typedef(nested)

        assert result == {
            "tag": "list",
            "element": {"tag": "list", "element": {"tag": "int"}},
        }

    def test_serialize_union_typedef(self) -> None:
        """Test serializing UnionType."""
        adapter = JSONAdapter()
        union = UnionType(options=(IntType(), StrType()))
        result = adapter.serialize_typedef(union)

        assert result == {
            "tag": "union",
            "options": [{"tag": "int"}, {"tag": "str"}],
        }

    def test_serialize_node_typedef(self) -> None:
        """Test serializing NodeType."""
        adapter = JSONAdapter()
        node_type = NodeType(returns=FloatType())
        result = adapter.serialize_typedef(node_type)

        assert result == {"tag": "node", "returns": {"tag": "float"}}

    def test_serialize_ref_typedef(self) -> None:
        """Test serializing RefType."""
        adapter = JSONAdapter()
        ref_type = RefType(target=NodeType(returns=IntType()))
        result = adapter.serialize_typedef(ref_type)

        assert result == {
            "tag": "ref",
            "target": {"tag": "node", "returns": {"tag": "int"}},
        }


class TestJSONAdapterDeserializeTypeDef:
    """Test JSONAdapter.deserialize_typedef() method."""

    def test_deserialize_primitive_typedef(self) -> None:
        """Test deserializing primitive TypeDef."""
        adapter = JSONAdapter()

        int_result = adapter.deserialize_typedef({"tag": "int"})
        assert isinstance(int_result, IntType)

        float_result = adapter.deserialize_typedef({"tag": "float"})
        assert isinstance(float_result, FloatType)

        str_result = adapter.deserialize_typedef({"tag": "str"})
        assert isinstance(str_result, StrType)

    def test_deserialize_list_typedef(self) -> None:
        """Test deserializing ListType."""
        adapter = JSONAdapter()
        data = {"tag": "list", "element": {"tag": "int"}}
        result = adapter.deserialize_typedef(data)

        assert isinstance(result, ListType)
        assert isinstance(result.element, IntType)

    def test_deserialize_dict_typedef(self) -> None:
        """Test deserializing DictType."""
        adapter = JSONAdapter()
        data = {"tag": "dict", "key": {"tag": "str"}, "value": {"tag": "float"}}
        result = adapter.deserialize_typedef(data)

        assert isinstance(result, DictType)
        assert isinstance(result.key, StrType)
        assert isinstance(result.value, FloatType)

    def test_deserialize_nested_typedef(self) -> None:
        """Test deserializing nested TypeDef."""
        adapter = JSONAdapter()
        data = {"tag": "list", "element": {"tag": "list", "element": {"tag": "int"}}}
        result = adapter.deserialize_typedef(data)

        assert isinstance(result, ListType)
        assert isinstance(result.element, ListType)
        assert isinstance(result.element.element, IntType)

    def test_deserialize_unknown_typedef_tag_raises_error(self) -> None:
        """Test that unknown TypeDef tag raises ValueError."""
        adapter = JSONAdapter()
        data = {"tag": "nonexistent_type_tag"}

        with pytest.raises(ValueError, match="Unknown TypeDef tag"):
            adapter.deserialize_typedef(data)


class TestJSONAdapterSerializeNodeSchema:
    """Test JSONAdapter.serialize_node_schema() method."""

    def test_serialize_simple_node_schema(self) -> None:
        """Test serializing schema for simple node."""

        class SimpleNode(Node[int], tag="simple_schema"):
            value: int

        adapter = JSONAdapter()
        schema = node_schema(SimpleNode)
        result = adapter.serialize_node_schema(schema)

        assert result["tag"] == "simple_schema"
        assert result["type_params"] == []
        assert result["returns"] == {"tag": "int"}
        assert len(result["fields"]) == 1
        assert result["fields"][0]["name"] == "value"
        assert result["fields"][0]["type"] == {"tag": "int"}

    def test_serialize_node_schema_with_multiple_fields(self) -> None:
        """Test serializing schema with multiple fields."""

        class MultiField(Node[str], tag="multi_schema"):
            name: str
            count: int
            active: bool

        adapter = JSONAdapter()
        schema = node_schema(MultiField)
        result = adapter.serialize_node_schema(schema)

        assert result["tag"] == "multi_schema"
        assert len(result["fields"]) == 3

        field_names = [f["name"] for f in result["fields"]]
        assert "name" in field_names
        assert "count" in field_names
        assert "active" in field_names

    def test_serialize_node_schema_with_node_field(self) -> None:
        """Test serializing schema for node containing another node."""

        class Container(Node[int], tag="container_schema"):
            child: Node[int]

        adapter = JSONAdapter()
        schema = node_schema(Container)
        result = adapter.serialize_node_schema(schema)

        assert result["tag"] == "container_schema"
        assert len(result["fields"]) == 1
        assert result["fields"][0]["name"] == "child"
        assert result["fields"][0]["type"]["tag"] == "node"
        assert result["fields"][0]["type"]["returns"]["tag"] == "int"

    def test_serialize_generic_node_schema(self) -> None:
        """Test serializing schema for generic node."""

        class GenericNode[T](Node[T], tag="generic_schema"):
            value: T

        adapter = JSONAdapter()
        schema = node_schema(GenericNode)
        result = adapter.serialize_node_schema(schema)

        assert result["tag"] == "generic_schema"
        assert len(result["type_params"]) == 1
        assert result["type_params"][0]["tag"] == "typeparam"
        assert result["type_params"][0]["name"] == "T"


class TestJSONAdapterRoundTrip:
    """Test round-trip serialization through JSONAdapter."""

    def test_node_round_trip(self) -> None:
        """Test node serialization round-trip."""

        class Data(Node[str], tag="data_adapter_rt"):
            text: str
            number: int

        adapter = JSONAdapter()
        original = Data(text="hello", number=42)

        serialized = adapter.serialize_node(original)
        deserialized = adapter.deserialize_node(serialized)

        assert deserialized == original

    def test_nested_node_round_trip(self) -> None:
        """Test nested node round-trip."""

        class Inner(Node[int], tag="inner_adapter_rt"):
            value: int

        class Outer(Node[int], tag="outer_adapter_rt"):
            child: Node[int]

        adapter = JSONAdapter()
        original = Outer(child=Inner(value=100))

        serialized = adapter.serialize_node(original)
        deserialized = adapter.deserialize_node(serialized)

        assert deserialized == original

    def test_typedef_round_trip(self) -> None:
        """Test TypeDef round-trip."""
        adapter = JSONAdapter()
        original = DictType(key=StrType(), value=ListType(element=IntType()))

        serialized = adapter.serialize_typedef(original)
        deserialized = adapter.deserialize_typedef(serialized)

        assert deserialized == original

    def test_complex_structure_round_trip(self) -> None:
        """Test complex nested structure round-trip."""

        class Const(Node[float], tag="const_adapter_rt"):
            value: float

        class BinOp(Node[float], tag="binop_adapter_rt"):
            op: str
            left: Node[float]
            right: Node[float]

        adapter = JSONAdapter()
        # Build: (1.0 + 2.0) * 3.0
        original = BinOp(
            op="*",
            left=BinOp(op="+", left=Const(value=1.0), right=Const(value=2.0)),
            right=Const(value=3.0),
        )

        serialized = adapter.serialize_node(original)
        deserialized = adapter.deserialize_node(serialized)

        assert deserialized == original


class TestJSONAdapterPrivateMethods:
    """Test private helper methods of JSONAdapter."""

    def test_serialize_value_with_nested_nodes(self) -> None:
        """Test _serialize_value handles nested nodes correctly."""

        class Num(Node[int], tag="num_private"):
            value: int

        adapter = JSONAdapter()
        node = Num(value=42)

        # _serialize_value is called internally, test through serialize_node
        result = adapter.serialize_node(node)
        assert result["value"] == 42

    def test_serialize_value_with_list_of_nodes(self) -> None:
        """Test serializing list containing nodes."""

        class Item(Node[int], tag="item_list_private"):
            value: int

        class Collection(Node[list[int]], tag="collection_private"):
            items: list[Node[int]]

        adapter = JSONAdapter()
        collection = Collection(items=[Item(value=1), Item(value=2)])
        result = adapter.serialize_node(collection)

        assert result["items"] == [
            {"tag": "item_list_private", "value": 1},
            {"tag": "item_list_private", "value": 2},
        ]

    def test_deserialize_value_with_nested_dicts(self) -> None:
        """Test _deserialize_value handles nested dicts correctly."""

        class Nested(Node[dict[str, int]], tag="nested_dict_private"):
            data: dict[str, int]

        adapter = JSONAdapter()
        data = {"tag": "nested_dict_private", "data": {"a": 1, "b": 2}}
        result = adapter.deserialize_node(data)

        assert result.data == {"a": 1, "b": 2}


class TestJSONAdapterEdgeCases:
    """Test edge cases and error conditions."""

    def test_serialize_node_with_none_field(self) -> None:
        """Test serializing node with None value."""

        class OptionalNode(Node[str], tag="optional_edge"):
            value: str | None

        adapter = JSONAdapter()
        node = OptionalNode(value=None)
        result = adapter.serialize_node(node)

        assert result == {"tag": "optional_edge", "value": None}

    def test_serialize_node_with_empty_collections(self) -> None:
        """Test serializing node with empty list and dict."""

        class EmptyNode(Node[str], tag="empty_edge"):
            items: list[int]
            data: dict[str, int]

        adapter = JSONAdapter()
        node = EmptyNode(items=[], data={})
        result = adapter.serialize_node(node)

        assert result == {"tag": "empty_edge", "items": [], "data": {}}

    def test_deserialize_preserves_types(self) -> None:
        """Test that deserialization preserves Python types correctly."""

        class TypeNode(Node[str], tag="type_edge"):
            int_val: int
            float_val: float
            bool_val: bool
            str_val: str

        adapter = JSONAdapter()
        data = {
            "tag": "type_edge",
            "int_val": 42,
            "float_val": 3.14,
            "bool_val": True,
            "str_val": "hello",
        }
        result = adapter.deserialize_node(data)

        assert isinstance(result.int_val, int)
        assert isinstance(result.float_val, float)
        assert isinstance(result.bool_val, bool)
        assert isinstance(result.str_val, str)
