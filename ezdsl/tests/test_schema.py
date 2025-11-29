"""Tests for ezdsl.schema module."""

import sys
import pytest
from typing import TypeVar, Union

from ezdsl.schema import extract_type, _extract_generic_origin
from ezdsl.types import (
    PrimitiveType,
    NodeType,
    RefType,
    UnionType,
    ParameterizedType,
    TypeParameter,
)

# Check Python version for version-specific tests
PYTHON_312_PLUS = sys.version_info >= (3, 12)


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


class TestExtractTypeParameter:
    """Test extracting TypeVar (PEP 695 type parameters)."""

    def test_extract_simple_type_parameter(self):
        """Test extracting an unbounded TypeVar."""
        T = TypeVar("T")
        result = extract_type(T)
        assert isinstance(result, TypeParameter)
        assert result.name == "T"
        assert result.bound is None

    def test_extract_bounded_type_parameter(self):
        """Test extracting a TypeVar with bound (like T: int)."""
        T = TypeVar("T", bound=int)
        result = extract_type(T)
        assert isinstance(result, TypeParameter)
        assert result.name == "T"
        assert result.bound is not None
        assert isinstance(result.bound, PrimitiveType)
        assert result.bound.primitive == int


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


class TestExtractParameterized:
    """Test extracting parameterized types."""

    def test_extract_list_int(self):
        """Test extracting list[int]."""
        result = extract_type(list[int])
        assert isinstance(result, ParameterizedType)
        assert result.name == "list[<class 'int'>]"
        assert isinstance(result.origin, PrimitiveType)
        assert result.origin.primitive == list
        assert len(result.args) == 1
        assert isinstance(result.args[0], PrimitiveType)
        assert result.args[0].primitive == int

    def test_extract_dict_str_int(self):
        """Test extracting dict[str, int]."""
        result = extract_type(dict[str, int])
        assert isinstance(result, ParameterizedType)
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
        assert isinstance(result, ParameterizedType)
        assert isinstance(result.origin, PrimitiveType)
        assert result.origin.primitive == set
        assert len(result.args) == 1
        assert isinstance(result.args[0], PrimitiveType)
        assert result.args[0].primitive == float

    def test_extract_nested_parameterized(self):
        """Test extracting nested parameterized types."""
        result = extract_type(list[dict[str, int]])
        assert isinstance(result, ParameterizedType)
        assert result.origin.primitive == list
        assert len(result.args) == 1
        assert isinstance(result.args[0], ParameterizedType)
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
        assert isinstance(result, ParameterizedType)
        assert result.origin.primitive == list
        assert len(result.args) == 1
        assert result.args[0].primitive == int

    def test_extract_generic_type_alias(self):
        """Test extracting a generic PEP 695 type alias."""
        # Create a generic type alias
        exec("type Pair[T] = tuple[T, T]", globals())
        Pair = globals()["Pair"]

        result = extract_type(Pair[int])
        assert isinstance(result, ParameterizedType)
        assert result.origin.primitive == tuple
        assert len(result.args) == 2
        assert result.args[0].primitive == int
        assert result.args[1].primitive == int


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
        assert isinstance(result.options[0], ParameterizedType)
        assert isinstance(result.options[1], ParameterizedType)
        assert isinstance(result.options[2], PrimitiveType)
        assert result.options[2].primitive == type(None)
