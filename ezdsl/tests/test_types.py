"""Tests for ezdsl.types module."""

import pytest

from ezdsl.types import (
    TypeDef,
    PrimitiveType,
    NodeType,
    RefType,
    UnionType,
    ParameterizedType,
    TypeParameter,
    PRIMITIVES,
)


class TestPrimitives:
    """Test primitive type handling."""

    def test_primitives_frozenset(self):
        """Test that PRIMITIVES contains expected types."""
        assert float in PRIMITIVES
        assert int in PRIMITIVES
        assert str in PRIMITIVES
        assert bool in PRIMITIVES
        assert type(None) in PRIMITIVES

    def test_primitives_immutable(self):
        """Test that PRIMITIVES is immutable."""
        with pytest.raises((TypeError, AttributeError)):
            PRIMITIVES.add(list)


class TestPrimitiveType:
    """Test PrimitiveType."""

    def test_primitive_type_creation(self):
        """Test creating a PrimitiveType."""
        pt = PrimitiveType(int)
        assert pt.primitive == int
        assert pt._tag == "primitive"

    def test_primitive_type_frozen(self):
        """Test that PrimitiveType is immutable."""
        pt = PrimitiveType(int)
        with pytest.raises((AttributeError, TypeError)):
            pt.primitive = float


class TestNodeType:
    """Test NodeType."""

    def test_node_type_creation(self):
        """Test creating a NodeType."""
        returns = PrimitiveType(float)
        nt = NodeType(returns)
        assert nt.returns == returns
        assert nt._tag == "node"

    def test_node_type_frozen(self):
        """Test that NodeType is immutable."""
        nt = NodeType(PrimitiveType(int))
        with pytest.raises((AttributeError, TypeError)):
            nt.returns = PrimitiveType(float)


class TestRefType:
    """Test RefType."""

    def test_ref_type_creation(self):
        """Test creating a RefType."""
        target = PrimitiveType(int)
        rt = RefType(target)
        assert rt.target == target
        assert rt._tag == "ref"

    def test_ref_type_frozen(self):
        """Test that RefType is immutable."""
        rt = RefType(PrimitiveType(int))
        with pytest.raises((AttributeError, TypeError)):
            rt.target = PrimitiveType(float)


class TestUnionType:
    """Test UnionType."""

    def test_union_type_creation(self):
        """Test creating a UnionType."""
        options = (PrimitiveType(int), PrimitiveType(str))
        ut = UnionType(options)
        assert ut.options == options
        assert ut._tag == "union"

    def test_union_type_frozen(self):
        """Test that UnionType is immutable."""
        ut = UnionType((PrimitiveType(int), PrimitiveType(str)))
        with pytest.raises((AttributeError, TypeError)):
            ut.options = (PrimitiveType(float),)


class TestParameterizedType:
    """Test ParameterizedType."""

    def test_parameterized_type_creation(self):
        """Test creating a ParameterizedType."""
        origin = PrimitiveType(list)
        args = (PrimitiveType(int),)
        pt = ParameterizedType(name="list[int]", origin=origin, args=args)
        assert pt.name == "list[int]"
        assert pt.origin == origin
        assert pt.args == args
        assert pt._tag == "parameterized"

    def test_parameterized_type_frozen(self):
        """Test that ParameterizedType is immutable."""
        pt = ParameterizedType(
            name="list[int]",
            origin=PrimitiveType(list),
            args=(PrimitiveType(int),)
        )
        with pytest.raises((AttributeError, TypeError)):
            pt.name = "list[str]"


class TestTypeParameter:
    """Test TypeParameter."""

    def test_type_parameter_basic(self):
        """Test creating a basic TypeParameter (unbounded)."""
        tp = TypeParameter(name="T")
        assert tp.name == "T"
        assert tp.bound is None
        assert tp._tag == "param"

    def test_type_parameter_with_bound(self):
        """Test TypeParameter with bound (like T: int)."""
        bound = PrimitiveType(int)
        tp = TypeParameter(name="T", bound=bound)
        assert tp.name == "T"
        assert tp.bound == bound

    def test_type_parameter_frozen(self):
        """Test that TypeParameter is immutable."""
        tp = TypeParameter(name="T")
        with pytest.raises((AttributeError, TypeError)):
            tp.name = "U"


class TestTypeDefRegistry:
    """Test TypeDef registry functionality."""

    def test_type_registry_contains_types(self):
        """Test that type registry contains all type definitions."""
        assert "primitive" in TypeDef._registry
        assert "node" in TypeDef._registry
        assert "ref" in TypeDef._registry
        assert "union" in TypeDef._registry
        assert "parameterized" in TypeDef._registry
        assert "param" in TypeDef._registry

    def test_type_registry_maps_to_classes(self):
        """Test that registry maps tags to correct classes."""
        assert TypeDef._registry["primitive"] == PrimitiveType
        assert TypeDef._registry["node"] == NodeType
        assert TypeDef._registry["ref"] == RefType
        assert TypeDef._registry["union"] == UnionType
        assert TypeDef._registry["parameterized"] == ParameterizedType
        assert TypeDef._registry["param"] == TypeParameter
