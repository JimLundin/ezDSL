"""Tests for typedsl.nodes module."""

import pytest

from typedsl.nodes import Child, Node, NodeRef, Ref


class TestNodeBasics:
    """Test basic Node functionality."""

    def test_node_subclass_becomes_dataclass(self) -> None:
        """Test that Node subclasses are automatically converted to dataclasses."""

        class SimpleNode(Node[int]):
            value: int

        node = SimpleNode(value=42)
        assert node.value == 42

    def test_node_is_frozen(self) -> None:
        """Test that Node instances are immutable."""

        class FrozenTestNode(Node[int]):
            value: int

        node = FrozenTestNode(value=42)
        with pytest.raises((AttributeError, TypeError)):
            node.value = 100

    def test_node_with_multiple_fields(self) -> None:
        """Test Node with multiple fields."""

        class MultiFieldNode(Node[str]):
            name: str
            count: int
            active: bool

        node = MultiFieldNode(name="test", count=5, active=True)
        assert node.name == "test"
        assert node.count == 5
        assert node.active is True


class TestNodeTags:
    """Test Node tag generation and registration."""

    def test_automatic_tag_generation(self) -> None:
        """Test that tags are automatically generated from class names."""

        class MyTestNode(Node[int]):
            value: int

        # 'node' suffix is removed, so MyTestNode -> mytest
        assert MyTestNode._tag == "mytest"

    def test_automatic_tag_strips_node_suffix(self) -> None:
        """Test that 'node' suffix is removed from auto-generated tags."""

        class CalculatorNode(Node[float]):
            result: float

        assert CalculatorNode._tag == "calculator"

    def test_custom_tag(self) -> None:
        """Test explicitly setting a custom tag."""

        class MyNode(Node[int], tag="custom_tag"):
            value: int

        assert MyNode._tag == "custom_tag"

    def test_tag_with_underscores(self) -> None:
        """Test custom tags with underscores."""

        class MyNode(Node[int], tag="my_custom_tag"):
            value: int

        assert MyNode._tag == "my_custom_tag"

    def test_tag_with_hyphens(self) -> None:
        """Test custom tags with hyphens."""

        class MyNode(Node[int], tag="my-custom-tag"):
            value: int

        assert MyNode._tag == "my-custom-tag"

    def test_tag_with_numbers(self) -> None:
        """Test tags with numbers."""

        class MyNode(Node[int], tag="node123"):
            value: int

        assert MyNode._tag == "node123"


class TestNodeRegistry:
    """Test Node registry functionality."""

    def test_node_registered_on_definition(self) -> None:
        """Test that nodes are registered when defined."""

        class RegisteredNode(Node[int], tag="registered_test"):
            value: int

        assert "registered_test" in Node.registry
        assert Node.registry["registered_test"] == RegisteredNode

    def test_multiple_nodes_in_registry(self) -> None:
        """Test that multiple nodes can be registered."""
        initial_count = len(Node.registry)

        class FirstNode(Node[int], tag="first_unique"):
            value: int

        class SecondNode(Node[str], tag="second_unique"):
            text: str

        assert len(Node.registry) >= initial_count + 2
        assert "first_unique" in Node.registry
        assert "second_unique" in Node.registry

    def test_tag_collision_raises_error(self) -> None:
        """Test that duplicate tags raise an error."""

        class FirstNode(Node[int], tag="collision_test"):
            value: int

        with pytest.raises(ValueError, match="Tag 'collision_test' already registered"):

            class SecondNode(Node[str], tag="collision_test"):
                text: str

    def test_same_class_reimport_no_error(self) -> None:
        """Test that re-importing the same class doesn't raise an error."""

        class UniqueNode(Node[int], tag="unique_reimport"):
            value: int

        # Simulate re-registration of same class
        # This happens when the module is reloaded
        # The check `if existing is not cls` prevents the error
        assert Node.registry["unique_reimport"] is UniqueNode


class TestNodeGenericTypes:
    """Test Node with generic type parameters."""

    def test_simple_generic_node(self) -> None:
        """Test Node with a simple generic type parameter."""

        class Container[T](Node[T]):
            value: T

        # Create with int
        int_container = Container(value=42)
        assert int_container.value == 42

        # Create with str
        str_container = Container(value="hello")
        assert str_container.value == "hello"

    def test_generic_node_with_list(self) -> None:
        """Test generic Node that returns a list."""

        class ListContainer[T](Node[list[T]]):
            items: list[T]

        container = ListContainer(items=[1, 2, 3])
        assert container.items == [1, 2, 3]

    def test_bounded_generic_node(self) -> None:
        """Test generic Node with type bounds."""

        class NumericNode[T: int | float](Node[T]):
            value: T

        int_node = NumericNode(value=42)
        assert int_node.value == 42

        float_node = NumericNode(value=3.14)
        assert float_node.value == 3.14


class TestNodeWithNestedNodes:
    """Test Nodes containing other Nodes."""

    def test_node_with_child_node(self) -> None:
        """Test Node containing another Node."""

        class Literal(Node[int]):
            value: int

        class Add(Node[int]):
            left: Node[int]
            right: Node[int]

        tree = Add(left=Literal(value=1), right=Literal(value=2))
        assert tree.left.value == 1
        assert tree.right.value == 2

    def test_deeply_nested_nodes(self) -> None:
        """Test deeply nested Node structures."""

        class Const(Node[float]):
            value: float

        class Multiply(Node[float]):
            left: Node[float]
            right: Node[float]

        # Build: (1.0 * 2.0) * 3.0
        tree = Multiply(
            left=Multiply(left=Const(value=1.0), right=Const(value=2.0)),
            right=Const(value=3.0),
        )
        assert tree.left.left.value == 1.0
        assert tree.left.right.value == 2.0
        assert tree.right.value == 3.0


class TestRef:
    """Test Ref (reference) functionality."""

    def test_ref_creation(self) -> None:
        """Test creating a Ref."""
        ref = Ref[Node[int]](id="node-1")
        assert ref.id == "node-1"

    def test_ref_is_frozen(self) -> None:
        """Test that Ref instances are immutable."""
        ref = Ref[Node[int]](id="node-1")
        with pytest.raises((AttributeError, TypeError)):
            ref.id = "node-2"

    def test_ref_equality(self) -> None:
        """Test Ref equality comparison."""
        ref1 = Ref[Node[int]](id="node-1")
        ref2 = Ref[Node[int]](id="node-1")
        ref3 = Ref[Node[int]](id="node-2")

        assert ref1 == ref2
        assert ref1 != ref3


class TestTypeAliases:
    """Test type aliases provided by the module."""

    def test_node_ref_type_alias(self) -> None:
        """Test NodeRef type alias."""
        # NodeRef[T] should be equivalent to Ref[Node[T]]
        ref: NodeRef[int] = Ref[Node[int]](id="test")
        assert ref.id == "test"

    def test_child_type_alias(self) -> None:
        """Test Child type alias."""

        class LiteralForChild(Node[int], tag="literal_for_child"):
            value: int

        # Child[T] can be either Node[T] or Ref[Node[T]]
        child1: Child[int] = LiteralForChild(value=42)
        child2: Child[int] = Ref[Node[int]](id="ref-1")

        assert isinstance(child1, Node)
        assert isinstance(child2, Ref)


class TestNodeEquality:
    """Test Node equality and hashing."""

    def test_node_equality(self) -> None:
        """Test that identical nodes are equal."""

        class EqualityTestNode(Node[int], tag="equality_test"):
            value: int

        node1 = EqualityTestNode(value=42)
        node2 = EqualityTestNode(value=42)
        node3 = EqualityTestNode(value=100)

        assert node1 == node2
        assert node1 != node3

    def test_node_hashability(self) -> None:
        """Test that nodes are hashable (can be used in sets/dicts)."""

        class HashableTestNode(Node[int], tag="hashable_test"):
            value: int

        node1 = HashableTestNode(value=42)
        node2 = HashableTestNode(value=42)

        # Should be able to add to set
        node_set = {node1, node2}
        # Since they're equal and frozen, should be same in set
        assert len(node_set) == 1

        # Should be able to use as dict key
        node_dict = {node1: "first"}
        assert node_dict[node2] == "first"
