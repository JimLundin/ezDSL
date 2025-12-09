"""Tests for typedsl.types module."""

import pytest

from typedsl.types import (
    BoolType,
    DateTimeType,
    DateType,
    DictType,
    DurationType,
    ExternalType,
    FloatType,
    IntType,
    ListType,
    LiteralType,
    MappingType,
    NodeType,
    NoneType,
    RefType,
    SequenceType,
    SetType,
    StrType,
    TimeType,
    TupleType,
    TypeDef,
    TypeParameter,
    TypeParameterRef,
    UnionType,
)


class TestPrimitiveTypes:
    """Test concrete primitive types."""

    def test_int_type_creation(self) -> None:
        """Test creating an IntType."""
        it = IntType()
        assert it._tag == "int"

    def test_float_type_creation(self) -> None:
        """Test creating a FloatType."""
        ft = FloatType()
        assert ft._tag == "float"

    def test_str_type_creation(self) -> None:
        """Test creating a StrType."""
        st = StrType()
        assert st._tag == "str"

    def test_bool_type_creation(self) -> None:
        """Test creating a BoolType."""
        bt = BoolType()
        assert bt._tag == "bool"

    def test_none_type_creation(self) -> None:
        """Test creating a NoneType."""
        nt = NoneType()
        assert nt._tag == "none"

    def test_primitive_types_frozen(self) -> None:
        """Test that primitive types are immutable."""
        it = IntType()
        with pytest.raises((AttributeError, TypeError)):
            it._tag = "other"


class TestTemporalTypes:
    """Test temporal types."""

    def test_date_type_creation(self) -> None:
        """Test creating a DateType."""
        dt = DateType()
        assert dt._tag == "date"

    def test_time_type_creation(self) -> None:
        """Test creating a TimeType."""
        tt = TimeType()
        assert tt._tag == "time"

    def test_datetime_type_creation(self) -> None:
        """Test creating a DateTimeType."""
        dtt = DateTimeType()
        assert dtt._tag == "datetime"

    def test_duration_type_creation(self) -> None:
        """Test creating a DurationType."""
        durt = DurationType()
        assert durt._tag == "duration"

    def test_temporal_types_frozen(self) -> None:
        """Test that temporal types are immutable."""
        dt = DateType()
        with pytest.raises((AttributeError, TypeError)):
            dt._tag = "other"


class TestContainerTypes:
    """Test concrete container types."""

    def test_list_type_creation(self) -> None:
        """Test creating a ListType."""
        lt = ListType(element=IntType())
        assert lt._tag == "list"
        assert isinstance(lt.element, IntType)

    def test_list_type_frozen(self) -> None:
        """Test that ListType is immutable."""
        lt = ListType(element=IntType())
        with pytest.raises((AttributeError, TypeError)):
            lt.element = StrType()

    def test_dict_type_creation(self) -> None:
        """Test creating a DictType."""
        dt = DictType(key=StrType(), value=IntType())
        assert dt._tag == "dict"
        assert isinstance(dt.key, StrType)
        assert isinstance(dt.value, IntType)

    def test_dict_type_frozen(self) -> None:
        """Test that DictType is immutable."""
        dt = DictType(key=StrType(), value=IntType())
        with pytest.raises((AttributeError, TypeError)):
            dt.key = IntType()


class TestGenericContainerTypes:
    """Test generic container types (Sequence, Mapping)."""

    def test_sequence_type_creation(self) -> None:
        """Test creating a SequenceType."""
        st = SequenceType(element=IntType())
        assert st._tag == "sequence"
        assert isinstance(st.element, IntType)

    def test_sequence_type_with_nested_type(self) -> None:
        """Test creating SequenceType with nested type."""
        st = SequenceType(element=ListType(element=StrType()))
        assert st._tag == "sequence"
        assert isinstance(st.element, ListType)
        assert isinstance(st.element.element, StrType)

    def test_sequence_type_frozen(self) -> None:
        """Test that SequenceType is immutable."""
        st = SequenceType(element=IntType())
        with pytest.raises((AttributeError, TypeError)):
            st.element = StrType()

    def test_mapping_type_creation(self) -> None:
        """Test creating a MappingType."""
        mt = MappingType(key=StrType(), value=IntType())
        assert mt._tag == "mapping"
        assert isinstance(mt.key, StrType)
        assert isinstance(mt.value, IntType)

    def test_mapping_type_with_nested_types(self) -> None:
        """Test creating MappingType with nested types."""
        mt = MappingType(key=StrType(), value=ListType(element=FloatType()))
        assert mt._tag == "mapping"
        assert isinstance(mt.key, StrType)
        assert isinstance(mt.value, ListType)
        assert isinstance(mt.value.element, FloatType)

    def test_mapping_type_frozen(self) -> None:
        """Test that MappingType is immutable."""
        mt = MappingType(key=StrType(), value=IntType())
        with pytest.raises((AttributeError, TypeError)):
            mt.key = IntType()


class TestNodeType:
    """Test NodeType."""

    def test_node_type_creation(self) -> None:
        """Test creating a NodeType."""
        nt = NodeType(returns=FloatType())
        assert nt.returns._tag == "float"
        assert nt._tag == "node"

    def test_node_type_frozen(self) -> None:
        """Test that NodeType is immutable."""
        nt = NodeType(returns=IntType())
        with pytest.raises((AttributeError, TypeError)):
            nt.returns = FloatType()


class TestRefType:
    """Test RefType."""

    def test_ref_type_creation(self) -> None:
        """Test creating a RefType."""
        rt = RefType(target=IntType())
        assert isinstance(rt.target, IntType)
        assert rt._tag == "ref"

    def test_ref_type_frozen(self) -> None:
        """Test that RefType is immutable."""
        rt = RefType(target=IntType())
        with pytest.raises((AttributeError, TypeError)):
            rt.target = FloatType()


class TestUnionType:
    """Test UnionType."""

    def test_union_type_creation(self) -> None:
        """Test creating a UnionType."""
        ut = UnionType(options=(IntType(), StrType()))
        assert len(ut.options) == 2
        assert isinstance(ut.options[0], IntType)
        assert isinstance(ut.options[1], StrType)
        assert ut._tag == "union"

    def test_union_type_frozen(self) -> None:
        """Test that UnionType is immutable."""
        ut = UnionType((IntType(), StrType()))
        with pytest.raises((AttributeError, TypeError)):
            ut.options = (FloatType(),)


class TestTypeParameter:
    """Test TypeVar."""

    def test_type_parameter_basic(self) -> None:
        """Test creating a basic TypeVar (unbounded)."""
        tp = TypeParameter(name="T")
        assert tp.name == "T"
        assert tp.bound is None
        assert tp._tag == "typeparam"

    def test_type_parameter_with_bound(self) -> None:
        """Test TypeVar with bound (like T: int)."""
        bound = IntType()
        tp = TypeParameter(name="T", bound=bound)
        assert tp.name == "T"
        assert isinstance(tp.bound, IntType)

    def test_type_parameter_frozen(self) -> None:
        """Test that TypeVar is immutable."""
        tp = TypeParameter(name="T")
        with pytest.raises((AttributeError, TypeError)):
            tp.name = "U"


class TestTypeDefRegistry:
    """Test TypeDef registry functionality."""

    def test_type_registry_contains_types(self) -> None:
        """Test that type registry contains all type definitions."""
        assert "int" in TypeDef.registry
        assert "float" in TypeDef.registry
        assert "str" in TypeDef.registry
        assert "bool" in TypeDef.registry
        assert "none" in TypeDef.registry
        # Temporal types
        assert "date" in TypeDef.registry
        assert "time" in TypeDef.registry
        assert "datetime" in TypeDef.registry
        assert "duration" in TypeDef.registry
        # Container types
        assert "list" in TypeDef.registry
        assert "dict" in TypeDef.registry
        assert "set" in TypeDef.registry
        assert "tuple" in TypeDef.registry
        # Generic container types
        assert "sequence" in TypeDef.registry
        assert "mapping" in TypeDef.registry
        assert "literal" in TypeDef.registry
        assert "node" in TypeDef.registry
        assert "ref" in TypeDef.registry
        assert "union" in TypeDef.registry
        assert "typeparam" in TypeDef.registry
        assert "typeparamref" in TypeDef.registry

    def test_type_registry_maps_to_classes(self) -> None:
        """Test that registry maps tags to correct classes."""
        assert TypeDef.registry["int"] == IntType
        assert TypeDef.registry["float"] == FloatType
        assert TypeDef.registry["str"] == StrType
        assert TypeDef.registry["bool"] == BoolType
        assert TypeDef.registry["none"] == NoneType
        # Temporal types
        assert TypeDef.registry["date"] == DateType
        assert TypeDef.registry["time"] == TimeType
        assert TypeDef.registry["datetime"] == DateTimeType
        assert TypeDef.registry["duration"] == DurationType
        # Container types
        assert TypeDef.registry["list"] == ListType
        assert TypeDef.registry["dict"] == DictType
        assert TypeDef.registry["set"] == SetType
        assert TypeDef.registry["tuple"] == TupleType
        # Generic container types
        assert TypeDef.registry["sequence"] == SequenceType
        assert TypeDef.registry["mapping"] == MappingType
        assert TypeDef.registry["literal"] == LiteralType
        assert TypeDef.registry["node"] == NodeType
        assert TypeDef.registry["ref"] == RefType
        assert TypeDef.registry["union"] == UnionType
        assert TypeDef.registry["typeparam"] == TypeParameter
        assert TypeDef.registry["typeparamref"] == TypeParameterRef
        assert TypeDef.registry["external"] == ExternalType


class TestNestedTypes:
    """Test nested type structures."""

    def test_list_of_list(self) -> None:
        """Test creating list[list[int]]."""
        inner = ListType(element=IntType())
        outer = ListType(element=inner)
        assert isinstance(outer.element, ListType)
        assert isinstance(outer.element.element, IntType)

    def test_dict_of_lists(self) -> None:
        """Test creating dict[str, list[int]]."""
        dt = DictType(key=StrType(), value=ListType(element=IntType()))
        assert isinstance(dt.key, StrType)
        assert isinstance(dt.value, ListType)
        assert isinstance(dt.value.element, IntType)

    def test_union_of_containers(self) -> None:
        """Test creating list[int] | dict[str, float]."""
        ut = UnionType(
            options=(
                ListType(element=IntType()),
                DictType(key=StrType(), value=FloatType()),
            ),
        )
        assert len(ut.options) == 2
        assert isinstance(ut.options[0], ListType)
        assert isinstance(ut.options[1], DictType)
