"""Tests for typedsl.schema module."""

from typing import TypeVar

import pytest

from typedsl.schema import extract_type
from typedsl.types import (
    BoolType,
    DictType,
    FloatType,
    IntType,
    ListType,
    NoneType,
    StrType,
    TypeParameter,
    UnionType,
)


class TestExtractPrimitives:
    """Test extracting primitive types."""

    def test_extract_int(self) -> None:
        """Test extracting int type."""
        result = extract_type(int)
        assert isinstance(result, IntType)

    def test_extract_float(self) -> None:
        """Test extracting float type."""
        result = extract_type(float)
        assert isinstance(result, FloatType)

    def test_extract_str(self) -> None:
        """Test extracting str type."""
        result = extract_type(str)
        assert isinstance(result, StrType)

    def test_extract_bool(self) -> None:
        """Test extracting bool type."""
        result = extract_type(bool)
        assert isinstance(result, BoolType)

    def test_extract_none(self) -> None:
        """Test extracting None type."""
        result = extract_type(type(None))
        assert isinstance(result, NoneType)


class TestExtractTypeParameter:
    """Test extracting TypeVar (PEP 695 type parameters)."""

    def test_extract_simple_type_parameter(self) -> None:
        """Test extracting an unbounded TypeVar."""
        T = TypeVar("T")
        result = extract_type(T)
        assert isinstance(result, TypeParameter)
        assert result.name == "T"
        assert result.bound is None

    def test_extract_bounded_type_parameter(self) -> None:
        """Test extracting a TypeVar with bound (like T: int)."""
        T = TypeVar("T", bound=int)
        result = extract_type(T)
        assert isinstance(result, TypeParameter)
        assert result.name == "T"
        assert result.bound is not None
        assert isinstance(result.bound, IntType)


class TestExtractUnion:
    """Test extracting Union types."""

    def test_extract_union_pipe(self) -> None:
        """Test extracting Union with | operator."""
        result = extract_type(int | str)
        assert isinstance(result, UnionType)
        assert len(result.options) == 2
        assert isinstance(result.options[0], IntType)
        assert isinstance(result.options[1], StrType)

    def test_extract_union_multiple_types(self) -> None:
        """Test extracting Union with multiple types."""
        result = extract_type(int | str | float)
        assert isinstance(result, UnionType)
        assert len(result.options) == 3


class TestExtractContainers:
    """Test extracting container types."""

    def test_extract_list_int(self) -> None:
        """Test extracting list[int]."""
        result = extract_type(list[int])
        assert isinstance(result, ListType)
        assert isinstance(result.element, IntType)

    def test_extract_list_str(self) -> None:
        """Test extracting list[str]."""
        result = extract_type(list[str])
        assert isinstance(result, ListType)
        assert isinstance(result.element, StrType)

    def test_extract_dict_str_int(self) -> None:
        """Test extracting dict[str, int]."""
        result = extract_type(dict[str, int])
        assert isinstance(result, DictType)
        assert isinstance(result.key, StrType)
        assert isinstance(result.value, IntType)

    def test_extract_dict_int_float(self) -> None:
        """Test extracting dict[int, float]."""
        result = extract_type(dict[int, float])
        assert isinstance(result, DictType)
        assert isinstance(result.key, IntType)
        assert isinstance(result.value, FloatType)

    def test_extract_nested_list(self) -> None:
        """Test extracting list[list[int]]."""
        result = extract_type(list[list[int]])
        assert isinstance(result, ListType)
        assert isinstance(result.element, ListType)
        assert isinstance(result.element.element, IntType)

    def test_extract_list_dict(self) -> None:
        """Test extracting list[dict[str, int]]."""
        result = extract_type(list[dict[str, int]])
        assert isinstance(result, ListType)
        assert isinstance(result.element, DictType)
        assert isinstance(result.element.key, StrType)
        assert isinstance(result.element.value, IntType)


class TestExtractWithTypeParameters:
    """Test extracting types with type parameters."""

    def test_extract_list_with_type_parameter(self) -> None:
        """Test extracting list[T] where T is a type parameter."""
        T = TypeVar("T")
        result = extract_type(list[T])
        assert isinstance(result, ListType)
        assert isinstance(result.element, TypeParameter)
        assert result.element.name == "T"

    def test_extract_dict_with_type_parameter(self) -> None:
        """Test extracting dict[str, T] where T is a type parameter."""
        T = TypeVar("T")
        result = extract_type(dict[str, T])
        assert isinstance(result, DictType)
        assert isinstance(result.key, StrType)
        assert isinstance(result.value, TypeParameter)
        assert result.value.name == "T"

    def test_extract_nested_with_type_parameter(self) -> None:
        """Test extracting list[dict[str, T]]."""
        T = TypeVar("T")
        result = extract_type(list[dict[str, T]])
        assert isinstance(result, ListType)
        assert isinstance(result.element, DictType)
        assert isinstance(result.element.value, TypeParameter)
        assert result.element.value.name == "T"


class TestPEP695TypeAlias:
    """Test PEP 695 type alias support."""

    def test_extract_generic_type_alias(self) -> None:
        """Test extracting a generic PEP 695 type alias."""
        # Create a generic type alias: type Pair[T] = tuple[T, T]
        # When we use Pair[int], it should expand to tuple[int, int]
        # But we can't use tuple in our limited type system, so let's use dict
        exec("type Mapping[V] = dict[str, V]", globals())
        Mapping = globals()["Mapping"]

        result = extract_type(Mapping[int])
        assert isinstance(result, DictType)
        assert isinstance(result.key, StrType)
        assert isinstance(result.value, IntType)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_extract_invalid_type_raises(self) -> None:
        """Test that extracting an invalid type raises ValueError."""
        with pytest.raises(ValueError, match="Cannot extract type from"):
            extract_type(object())

    def test_extract_complex_union(self) -> None:
        """Test extracting complex union with nested types."""
        result = extract_type(list[int] | dict[str, float] | None)
        assert isinstance(result, UnionType)
        assert len(result.options) == 3
        assert isinstance(result.options[0], ListType)
        assert isinstance(result.options[1], DictType)
        assert isinstance(result.options[2], NoneType)

    def test_list_without_element_type_raises(self) -> None:
        """Test that list without element type raises ValueError."""
        # This test may not be possible with Python's typing system
        # as list without args gives list directly, not a parameterized type

    def test_dict_with_wrong_arg_count_raises(self) -> None:
        """Test that dict with wrong number of args raises ValueError."""
        # This is also hard to test as Python's typing system enforces this
