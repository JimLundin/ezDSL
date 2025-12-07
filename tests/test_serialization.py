"""Tests for typedsl.serialization module."""

import json

import pytest

from typedsl.nodes import Node, Ref
from typedsl.serialization import from_dict, from_json, to_dict, to_json
from typedsl.types import (
    FloatType,
    IntType,
    ListType,
    UnionType,
)


class TestToDictNode:
    """Test to_dict() with Node objects."""

    def test_simple_node_to_dict(self) -> None:
        """Test serializing a simple node to dict."""

        class Literal(Node[int], tag="literal_simple"):
            value: int

        node = Literal(value=42)
        result = to_dict(node)

        assert result == {"tag": "literal_simple", "value": 42}

    def test_node_with_multiple_fields_to_dict(self) -> None:
        """Test serializing node with multiple fields."""

        class Person(Node[str], tag="person_serial"):
            name: str
            age: int
            active: bool

        node = Person(name="Alice", age=30, active=True)
        result = to_dict(node)

        assert result == {
            "tag": "person_serial",
            "name": "Alice",
            "age": 30,
            "active": True,
        }

    def test_nested_nodes_to_dict(self) -> None:
        """Test serializing nested nodes."""

        class Const(Node[float], tag="const_nested"):
            value: float

        class Add(Node[float], tag="add_nested"):
            left: Node[float]
            right: Node[float]

        tree = Add(left=Const(value=1.5), right=Const(value=2.5))
        result = to_dict(tree)

        assert result == {
            "tag": "add_nested",
            "left": {"tag": "const_nested", "value": 1.5},
            "right": {"tag": "const_nested", "value": 2.5},
        }

    def test_deeply_nested_nodes_to_dict(self) -> None:
        """Test serializing deeply nested node structures."""

        class Val(Node[int], tag="val_deep"):
            value: int

        class Mul(Node[int], tag="mul_deep"):
            left: Node[int]
            right: Node[int]

        # Build: (2 * 3) * 4
        tree = Mul(left=Mul(left=Val(value=2), right=Val(value=3)), right=Val(value=4))
        result = to_dict(tree)

        assert result == {
            "tag": "mul_deep",
            "left": {
                "tag": "mul_deep",
                "left": {"tag": "val_deep", "value": 2},
                "right": {"tag": "val_deep", "value": 3},
            },
            "right": {"tag": "val_deep", "value": 4},
        }

    def test_node_with_list_field_to_dict(self) -> None:
        """Test serializing node with list field."""

        class ListNode(Node[list[int]], tag="list_node_serial"):
            items: list[int]

        node = ListNode(items=[1, 2, 3, 4, 5])
        result = to_dict(node)

        assert result == {"tag": "list_node_serial", "items": [1, 2, 3, 4, 5]}

    def test_node_with_dict_field_to_dict(self) -> None:
        """Test serializing node with dict field."""

        class DictNode(Node[dict[str, int]], tag="dict_node_serial"):
            data: dict[str, int]

        node = DictNode(data={"a": 1, "b": 2})
        result = to_dict(node)

        assert result == {"tag": "dict_node_serial", "data": {"a": 1, "b": 2}}


class TestToDictRef:
    """Test to_dict() with Ref objects."""

    def test_ref_to_dict(self) -> None:
        """Test serializing a Ref to dict."""
        ref = Ref[Node[int]](id="node-123")
        result = to_dict(ref)

        assert result == {"tag": "ref", "id": "node-123"}

    def test_multiple_refs_to_dict(self) -> None:
        """Test serializing multiple different Refs."""
        ref1 = Ref[Node[int]](id="alpha")
        ref2 = Ref[Node[str]](id="beta")

        result1 = to_dict(ref1)
        result2 = to_dict(ref2)

        assert result1 == {"tag": "ref", "id": "alpha"}
        assert result2 == {"tag": "ref", "id": "beta"}


class TestToDictTypeDef:
    """Test to_dict() with TypeDef objects."""

    def test_simple_typedef_to_dict(self) -> None:
        """Test serializing simple TypeDef."""
        int_type = IntType()
        result = to_dict(int_type)

        assert result == {"tag": "int"}

    def test_container_typedef_to_dict(self) -> None:
        """Test serializing container TypeDef."""
        list_type = ListType(element=FloatType())
        result = to_dict(list_type)

        assert result == {"tag": "list", "element": {"tag": "float"}}

    def test_nested_typedef_to_dict(self) -> None:
        """Test serializing nested TypeDef."""
        # list[list[int]]
        nested_list = ListType(element=ListType(element=IntType()))
        result = to_dict(nested_list)

        assert result == {
            "tag": "list",
            "element": {"tag": "list", "element": {"tag": "int"}},
        }

    def test_union_typedef_to_dict(self) -> None:
        """Test serializing UnionType."""
        union_type = UnionType(options=(IntType(), FloatType()))
        result = to_dict(union_type)

        assert result == {
            "tag": "union",
            "options": [{"tag": "int"}, {"tag": "float"}],
        }


class TestFromDict:
    """Test from_dict() deserialization."""

    def test_simple_node_from_dict(self) -> None:
        """Test deserializing a simple node from dict."""

        class Number(Node[int], tag="number_deser"):
            value: int

        data = {"tag": "number_deser", "value": 99}
        result = from_dict(data)

        assert isinstance(result, Number)
        assert result.value == 99

    def test_nested_nodes_from_dict(self) -> None:
        """Test deserializing nested nodes."""

        class Leaf(Node[str], tag="leaf_deser"):
            text: str

        class Branch(Node[str], tag="branch_deser"):
            left: Node[str]
            right: Node[str]

        data = {
            "tag": "branch_deser",
            "left": {"tag": "leaf_deser", "text": "hello"},
            "right": {"tag": "leaf_deser", "text": "world"},
        }
        result = from_dict(data)

        assert isinstance(result, Branch)
        assert isinstance(result.left, Leaf)
        assert isinstance(result.right, Leaf)
        assert result.left.text == "hello"
        assert result.right.text == "world"

    def test_ref_from_dict(self) -> None:
        """Test deserializing a Ref from dict."""
        data = {"tag": "ref", "id": "my-node-id"}
        result = from_dict(data)

        assert isinstance(result, Ref)
        assert result.id == "my-node-id"

    def test_typedef_from_dict(self) -> None:
        """Test deserializing TypeDef from dict."""
        data = {"tag": "int"}
        result = from_dict(data)

        assert isinstance(result, IntType)

    def test_complex_typedef_from_dict(self) -> None:
        """Test deserializing complex TypeDef."""
        data = {"tag": "list", "element": {"tag": "float"}}
        result = from_dict(data)

        assert isinstance(result, ListType)
        assert isinstance(result.element, FloatType)

    def test_unknown_tag_raises_error(self) -> None:
        """Test that unknown tag raises ValueError."""
        data = {"tag": "nonexistent_tag_xyz"}

        with pytest.raises(ValueError, match="Unknown tag"):
            from_dict(data)

    def test_missing_tag_raises_error(self) -> None:
        """Test that missing tag field raises KeyError with clear message."""
        data = {"value": 42}  # No 'tag' field

        with pytest.raises(KeyError, match="Missing required 'tag' field"):
            from_dict(data)

    def test_missing_ref_id_raises_error(self) -> None:
        """Test that missing id field for ref raises KeyError."""
        data = {"tag": "ref"}  # Missing 'id' field

        with pytest.raises(KeyError, match="Missing required 'id' field for ref"):
            from_dict(data)

    def test_unknown_tag_lists_available_tags(self) -> None:
        """Test that unknown tag error lists available tags."""
        data = {"tag": "completely_unknown_tag_xyz"}

        with pytest.raises(ValueError) as exc_info:
            from_dict(data)

        error_msg = str(exc_info.value)
        # Should mention the unknown tag
        assert "completely_unknown_tag_xyz" in error_msg
        # Should list available tags
        assert "Available node tags" in error_msg
        assert "Available typedef tags" in error_msg


class TestRoundTripSerialization:
    """Test round-trip serialization (to_dict -> from_dict)."""

    def test_simple_node_round_trip(self) -> None:
        """Test round-trip for simple node."""

        class Value(Node[int], tag="value_roundtrip"):
            num: int

        original = Value(num=123)
        serialized = to_dict(original)
        deserialized = from_dict(serialized)

        assert deserialized == original

    def test_nested_node_round_trip(self) -> None:
        """Test round-trip for nested nodes."""

        class Num(Node[float], tag="num_rt"):
            value: float

        class Sum(Node[float], tag="sum_rt"):
            a: Node[float]
            b: Node[float]

        original = Sum(a=Num(value=10.5), b=Num(value=20.5))
        serialized = to_dict(original)
        deserialized = from_dict(serialized)

        assert deserialized == original

    def test_ref_round_trip(self) -> None:
        """Test round-trip for Ref."""
        original = Ref[Node[str]](id="test-ref-id")
        serialized = to_dict(original)
        deserialized = from_dict(serialized)

        assert deserialized == original

    def test_typedef_round_trip(self) -> None:
        """Test round-trip for TypeDef."""
        original = ListType(element=IntType())
        serialized = to_dict(original)
        deserialized = from_dict(serialized)

        assert deserialized == original

    def test_complex_structure_round_trip(self) -> None:
        """Test round-trip for complex node structure."""

        class Literal(Node[int], tag="lit_complex_rt"):
            value: int

        class BinOp(Node[int], tag="binop_complex_rt"):
            op: str
            left: Node[int]
            right: Node[int]

        # Build: (1 + 2) * (3 + 4)
        original = BinOp(
            op="*",
            left=BinOp(op="+", left=Literal(value=1), right=Literal(value=2)),
            right=BinOp(op="+", left=Literal(value=3), right=Literal(value=4)),
        )

        serialized = to_dict(original)
        deserialized = from_dict(serialized)

        assert deserialized == original


class TestToJson:
    """Test to_json() JSON string serialization."""

    def test_simple_node_to_json(self) -> None:
        """Test serializing node to JSON string."""

        class Item(Node[str], tag="item_json"):
            name: str

        node = Item(name="test")
        result = to_json(node)

        # Should be valid JSON
        parsed = json.loads(result)
        assert parsed == {"tag": "item_json", "name": "test"}

    def test_json_is_formatted(self) -> None:
        """Test that JSON output is formatted (indented)."""

        class Simple(Node[int], tag="simple_json"):
            value: int

        node = Simple(value=42)
        result = to_json(node)

        # Should contain newlines (formatted)
        assert "\n" in result

    def test_nested_node_to_json(self) -> None:
        """Test serializing nested node to JSON."""

        class Inner(Node[int], tag="inner_json"):
            value: int

        class Outer(Node[int], tag="outer_json"):
            child: Node[int]

        node = Outer(child=Inner(value=100))
        result = to_json(node)

        parsed = json.loads(result)
        assert parsed == {
            "tag": "outer_json",
            "child": {"tag": "inner_json", "value": 100},
        }


class TestFromJson:
    """Test from_json() JSON string deserialization."""

    def test_simple_node_from_json(self) -> None:
        """Test deserializing node from JSON string."""

        class Product(Node[str], tag="product_json"):
            name: str
            price: float

        json_str = '{"tag": "product_json", "name": "Widget", "price": 9.99}'
        result = from_json(json_str)

        assert isinstance(result, Product)
        assert result.name == "Widget"
        assert result.price == 9.99

    def test_nested_node_from_json(self) -> None:
        """Test deserializing nested node from JSON."""

        class Point(Node[tuple[int, int]], tag="point_json"):
            x: int
            y: int

        class Line(Node[str], tag="line_json"):
            start: Node[tuple[int, int]]
            end: Node[tuple[int, int]]

        json_str = """{
            "tag": "line_json",
            "start": {"tag": "point_json", "x": 0, "y": 0},
            "end": {"tag": "point_json", "x": 10, "y": 10}
        }"""
        result = from_json(json_str)

        assert isinstance(result, Line)
        assert isinstance(result.start, Point)
        assert isinstance(result.end, Point)
        assert result.start.x == 0
        assert result.end.x == 10

    def test_invalid_json_raises_error(self) -> None:
        """Test that invalid JSON raises error."""
        invalid_json = "{this is not valid json"

        with pytest.raises(json.JSONDecodeError):
            from_json(invalid_json)

    def test_ref_from_json(self) -> None:
        """Test deserializing Ref from JSON."""
        json_str = '{"tag": "ref", "id": "node-999"}'
        result = from_json(json_str)

        assert isinstance(result, Ref)
        assert result.id == "node-999"


class TestJsonRoundTrip:
    """Test full JSON round-trip (to_json -> from_json)."""

    def test_simple_json_round_trip(self) -> None:
        """Test JSON round-trip for simple node."""

        class Data(Node[str], tag="data_json_rt"):
            content: str
            count: int

        original = Data(content="test data", count=42)
        json_str = to_json(original)
        deserialized = from_json(json_str)

        assert deserialized == original

    def test_complex_json_round_trip(self) -> None:
        """Test JSON round-trip for complex nested structure."""

        class Const(Node[float], tag="const_json_rt"):
            value: float

        class Expr(Node[float], tag="expr_json_rt"):
            operation: str
            left: Node[float]
            right: Node[float]

        original = Expr(
            operation="add",
            left=Const(value=1.5),
            right=Expr(operation="mul", left=Const(value=2.0), right=Const(value=3.0)),
        )

        json_str = to_json(original)
        deserialized = from_json(json_str)

        assert deserialized == original

    def test_typedef_json_round_trip(self) -> None:
        """Test JSON round-trip for TypeDef."""
        original = IntType()
        json_str = to_json(original)
        deserialized = from_json(json_str)

        assert deserialized == original


class TestSerializationEdgeCases:
    """Test edge cases in serialization."""

    def test_node_with_none_value(self) -> None:
        """Test serializing node with None field value."""

        class Optional(Node[str], tag="optional_serial"):
            value: str | None

        node = Optional(value=None)
        result = to_dict(node)

        assert result == {"tag": "optional_serial", "value": None}

        # Round trip
        deserialized = from_dict(result)
        assert deserialized == node

    def test_node_with_empty_list(self) -> None:
        """Test serializing node with empty list."""

        class Collection(Node[list[int]], tag="collection_serial"):
            items: list[int]

        node = Collection(items=[])
        result = to_dict(node)

        assert result == {"tag": "collection_serial", "items": []}

    def test_node_with_empty_dict(self) -> None:
        """Test serializing node with empty dict."""

        class Mapping(Node[dict[str, int]], tag="mapping_serial"):
            data: dict[str, int]

        node = Mapping(data={})
        result = to_dict(node)

        assert result == {"tag": "mapping_serial", "data": {}}

    def test_serialize_unsupported_type_raises_error(self) -> None:
        """Test that serializing unsupported type raises error."""

        class CustomClass:
            pass

        obj = CustomClass()

        with pytest.raises(ValueError, match="Cannot serialize"):
            to_dict(obj)  # type: ignore


class TestSerializationTypes:
    """Test serialization preserves type information."""

    def test_int_field_type_preserved(self) -> None:
        """Test that int fields remain ints after round-trip."""

        class IntNode(Node[int], tag="int_preserve"):
            value: int

        original = IntNode(value=42)
        result = from_dict(to_dict(original))

        assert isinstance(result.value, int)
        assert result.value == 42

    def test_float_field_type_preserved(self) -> None:
        """Test that float fields remain floats after round-trip."""

        class FloatNode(Node[float], tag="float_preserve"):
            value: float

        original = FloatNode(value=3.14)
        result = from_dict(to_dict(original))

        assert isinstance(result.value, float)
        assert result.value == 3.14

    def test_bool_field_type_preserved(self) -> None:
        """Test that bool fields remain bools after round-trip."""

        class BoolNode(Node[bool], tag="bool_preserve"):
            flag: bool

        original_true = BoolNode(flag=True)
        result_true = from_dict(to_dict(original_true))
        assert result_true.flag is True
        assert isinstance(result_true.flag, bool)

        original_false = BoolNode(flag=False)
        result_false = from_dict(to_dict(original_false))
        assert result_false.flag is False
        assert isinstance(result_false.flag, bool)
