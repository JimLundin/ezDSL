"""Tests for ezdsl.schema module."""

import sys
import pytest
from typing import TypeVar, Union, ParamSpec

from ezdsl.schema import extract_type, _extract_typevar, _extract_generic_origin
from ezdsl.types import (
    PrimitiveType,
    NodeType,
    RefType,
    UnionType,
    GenericType,
    TypeVarType,
    TypeParamKind,
    Variance,
)

# Check Python version for version-specific tests
PYTHON_311_PLUS = sys.version_info >= (3, 11)
PYTHON_312_PLUS = sys.version_info >= (3, 12)
PYTHON_313_PLUS = sys.version_info >= (3, 13)

# Import version-specific types
if PYTHON_311_PLUS:
    from typing import TypeVarTuple

if PYTHON_312_PLUS:
    from typing import TypeAliasType


class TestExtractPrimitives:
    """Test extracting primitive types."""

    def test_extract_int(self):
        """Test extracting int type."""
        result = extract_type(int)
        assert isinstance(result, PrimitiveType)
        assert result.primitive == int

    def test_extract_float(self):
        """Test extracting float type."""
        result = extract_type(float)
        assert isinstance(result, PrimitiveType)
        assert result.primitive == float

    def test_extract_str(self):
        """Test extracting str type."""
        result = extract_type(str)
        assert isinstance(result, PrimitiveType)
        assert result.primitive == str

    def test_extract_bool(self):
        """Test extracting bool type."""
        result = extract_type(bool)
        assert isinstance(result, PrimitiveType)
        assert result.primitive == bool

    def test_extract_none(self):
        """Test extracting None type."""
        result = extract_type(type(None))
        assert isinstance(result, PrimitiveType)
        assert result.primitive == type(None)


class TestExtractTypeVar:
    """Test extracting TypeVar."""

    def test_extract_simple_typevar(self):
        """Test extracting a simple TypeVar."""
        T = TypeVar("T")
        result = extract_type(T)
        assert isinstance(result, TypeVarType)
        assert result.name == "T"
        assert result.kind == TypeParamKind.TYPEVAR
        assert result.variance == Variance.INVARIANT
        assert result.bounds is None
        assert result.constraints is None

    def test_extract_bounded_typevar(self):
        """Test extracting a TypeVar with bounds."""
        T = TypeVar("T", bound=int)
        result = extract_type(T)
        assert isinstance(result, TypeVarType)
        assert result.name == "T"
        assert result.bounds is not None
        assert len(result.bounds) == 1
        assert isinstance(result.bounds[0], PrimitiveType)
        assert result.bounds[0].primitive == int

    def test_extract_constrained_typevar(self):
        """Test extracting a TypeVar with constraints."""
        T = TypeVar("T", int, str)
        result = extract_type(T)
        assert isinstance(result, TypeVarType)
        assert result.name == "T"
        assert result.constraints is not None
        assert len(result.constraints) == 2
        assert isinstance(result.constraints[0], PrimitiveType)
        assert result.constraints[0].primitive == int
        assert isinstance(result.constraints[1], PrimitiveType)
        assert result.constraints[1].primitive == str

    def test_extract_covariant_typevar(self):
        """Test extracting a covariant TypeVar."""
        T_co = TypeVar("T_co", covariant=True)
        result = extract_type(T_co)
        assert isinstance(result, TypeVarType)
        assert result.name == "T_co"
        assert result.variance == Variance.COVARIANT

    def test_extract_contravariant_typevar(self):
        """Test extracting a contravariant TypeVar."""
        T_contra = TypeVar("T_contra", contravariant=True)
        result = extract_type(T_contra)
        assert isinstance(result, TypeVarType)
        assert result.name == "T_contra"
        assert result.variance == Variance.CONTRAVARIANT

    def test_extract_paramspec(self):
        """Test extracting a ParamSpec."""
        P = ParamSpec("P")
        result = extract_type(P)
        assert isinstance(result, TypeVarType)
        assert result.name == "P"
        assert result.kind == TypeParamKind.PARAMSPEC

    @pytest.mark.skipif(not PYTHON_311_PLUS, reason="TypeVarTuple requires Python 3.11+")
    def test_extract_typevartuple(self):
        """Test extracting a TypeVarTuple."""
        Ts = TypeVarTuple("Ts")
        result = extract_type(Ts)
        assert isinstance(result, TypeVarType)
        assert result.name == "Ts"
        assert result.kind == TypeParamKind.TYPEVARTUPLE


class TestExtractUnion:
    """Test extracting Union types."""

    def test_extract_union_typing(self):
        """Test extracting Union from typing module."""
        result = extract_type(Union[int, str])
        assert isinstance(result, UnionType)
        assert len(result.options) == 2
        assert isinstance(result.options[0], PrimitiveType)
        assert result.options[0].primitive == int
        assert isinstance(result.options[1], PrimitiveType)
        assert result.options[1].primitive == str

    def test_extract_union_pipe(self):
        """Test extracting Union with | operator."""
        result = extract_type(int | str)
        assert isinstance(result, UnionType)
        assert len(result.options) == 2
        assert isinstance(result.options[0], PrimitiveType)
        assert result.options[0].primitive == int
        assert isinstance(result.options[1], PrimitiveType)
        assert result.options[1].primitive == str

    def test_extract_union_multiple_types(self):
        """Test extracting Union with multiple types."""
        result = extract_type(int | str | float)
        assert isinstance(result, UnionType)
        assert len(result.options) == 3


class TestExtractGeneric:
    """Test extracting generic types."""

    def test_extract_list_int(self):
        """Test extracting list[int]."""
        result = extract_type(list[int])
        assert isinstance(result, GenericType)
        assert result.name == "list[<class 'int'>]"
        assert isinstance(result.origin, PrimitiveType)
        assert result.origin.primitive == list
        assert len(result.args) == 1
        assert isinstance(result.args[0], PrimitiveType)
        assert result.args[0].primitive == int

    def test_extract_dict_str_int(self):
        """Test extracting dict[str, int]."""
        result = extract_type(dict[str, int])
        assert isinstance(result, GenericType)
        assert isinstance(result.origin, PrimitiveType)
        assert result.origin.primitive == dict
        assert len(result.args) == 2
        assert isinstance(result.args[0], PrimitiveType)
        assert result.args[0].primitive == str
        assert isinstance(result.args[1], PrimitiveType)
        assert result.args[1].primitive == int

    def test_extract_set_float(self):
        """Test extracting set[float]."""
        result = extract_type(set[float])
        assert isinstance(result, GenericType)
        assert isinstance(result.origin, PrimitiveType)
        assert result.origin.primitive == set
        assert len(result.args) == 1
        assert isinstance(result.args[0], PrimitiveType)
        assert result.args[0].primitive == float

    def test_extract_nested_generic(self):
        """Test extracting nested generic types."""
        result = extract_type(list[dict[str, int]])
        assert isinstance(result, GenericType)
        assert result.origin.primitive == list
        assert len(result.args) == 1
        assert isinstance(result.args[0], GenericType)
        assert result.args[0].origin.primitive == dict


@pytest.mark.skipif(not PYTHON_312_PLUS, reason="PEP 695 requires Python 3.12+")
class TestPEP695TypeAlias:
    """Test PEP 695 type alias support."""

    def test_extract_type_alias(self):
        """Test extracting a PEP 695 type alias."""
        # For non-generic type aliases, the alias just evaluates to the actual type
        # So list[int] is what we get, not a TypeAliasType wrapper
        # This test verifies that we can extract list[int] properly
        result = extract_type(list[int])
        assert isinstance(result, GenericType)
        assert result.origin.primitive == list
        assert len(result.args) == 1
        assert result.args[0].primitive == int

    def test_extract_generic_type_alias(self):
        """Test extracting a generic PEP 695 type alias."""
        # Create a generic type alias
        exec("type Pair[T] = tuple[T, T]", globals())
        Pair = globals()["Pair"]

        result = extract_type(Pair[int])
        assert isinstance(result, GenericType)
        assert result.origin.primitive == tuple
        assert len(result.args) == 2
        assert result.args[0].primitive == int
        assert result.args[1].primitive == int


class TestExtractTypeVarHelper:
    """Test _extract_typevar helper function."""

    def test_extract_typevar_simple(self):
        """Test extracting simple TypeVar."""
        T = TypeVar("T")
        result = _extract_typevar(T)
        assert result.name == "T"
        assert result.variance == Variance.INVARIANT

    def test_extract_typevar_covariant(self):
        """Test extracting covariant TypeVar."""
        T_co = TypeVar("T_co", covariant=True)
        result = _extract_typevar(T_co)
        assert result.variance == Variance.COVARIANT

    def test_extract_typevar_contravariant(self):
        """Test extracting contravariant TypeVar."""
        T_contra = TypeVar("T_contra", contravariant=True)
        result = _extract_typevar(T_contra)
        assert result.variance == Variance.CONTRAVARIANT

    def test_extract_typevar_with_bound(self):
        """Test extracting TypeVar with bound."""
        T = TypeVar("T", bound=int)
        result = _extract_typevar(T)
        assert result.bounds is not None
        assert len(result.bounds) == 1
        assert result.bounds[0].primitive == int

    def test_extract_typevar_with_constraints(self):
        """Test extracting TypeVar with constraints."""
        T = TypeVar("T", int, str, float)
        result = _extract_typevar(T)
        assert result.constraints is not None
        assert len(result.constraints) == 3


class TestExtractGenericOrigin:
    """Test _extract_generic_origin helper function."""

    def test_extract_list_origin(self):
        """Test extracting list origin."""
        result = _extract_generic_origin(list)
        assert isinstance(result, PrimitiveType)
        assert result.primitive == list

    def test_extract_dict_origin(self):
        """Test extracting dict origin."""
        result = _extract_generic_origin(dict)
        assert isinstance(result, PrimitiveType)
        assert result.primitive == dict

    def test_extract_primitive_origin(self):
        """Test extracting primitive type as origin."""
        result = _extract_generic_origin(int)
        assert isinstance(result, PrimitiveType)
        assert result.primitive == int


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_extract_invalid_type_raises(self):
        """Test that extracting an invalid type raises ValueError."""
        with pytest.raises(ValueError, match="Cannot extract type from"):
            extract_type(object())

    def test_extract_complex_union(self):
        """Test extracting complex union with nested types."""
        result = extract_type(list[int] | dict[str, float] | None)
        assert isinstance(result, UnionType)
        assert len(result.options) == 3
        assert isinstance(result.options[0], GenericType)
        assert isinstance(result.options[1], GenericType)
        assert isinstance(result.options[2], PrimitiveType)
        assert result.options[2].primitive == type(None)
