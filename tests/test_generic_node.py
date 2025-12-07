"""Tests for how generic Node definitions serialize their types."""

from typing import TypeVar

from typedsl.schema import extract_type
from typedsl.types import (
    DictType,
    IntType,
    ListType,
    StrType,
    TypeParameter,
)


def test_type_parameter_in_annotation() -> None:
    """Test extracting a type annotation that uses a type parameter."""
    # Simulate: class MyNode[T]:
    #               args: list[T]
    T = TypeVar("T")

    # What does list[T] extract to?
    result = extract_type(list[T])

    assert isinstance(result, ListType)

    # The element is the TypeVar T, not a concrete type!
    assert isinstance(result.element, TypeParameter)
    assert result.element.name == "T"


def test_nested_type_parameters() -> None:
    """Test extracting nested parameterized types with type parameters."""
    # Simulate: class MyNode[T]:
    #               args: list[dict[str, T]]
    T = TypeVar("T")

    result = extract_type(list[dict[str, T]])

    # Outer: list
    assert isinstance(result, ListType)

    # Middle: dict[str, T]
    dict_type = result.element
    assert isinstance(dict_type, DictType)

    # First arg of dict is str (concrete)
    assert isinstance(dict_type.key, StrType)

    # Second arg of dict is T (type parameter)
    assert isinstance(dict_type.value, TypeParameter)
    assert dict_type.value.name == "T"


def test_bounded_type_parameter_in_annotation() -> None:
    """Test extracting type annotations with bounded type parameters."""
    # Simulate: class MyNode[T: int]:
    #               value: T
    T = TypeVar("T", bound=int)

    result = extract_type(T)

    assert isinstance(result, TypeParameter)
    assert result.name == "T"
    assert result.bound is not None
    assert isinstance(result.bound, IntType)


def test_multiple_type_parameters() -> None:
    """Test multiple type parameters in one annotation."""
    # Simulate: class MyNode[K, V]:
    #               data: dict[K, V]
    K = TypeVar("K")
    V = TypeVar("V")

    result = extract_type(dict[K, V])

    assert isinstance(result, DictType)

    # Both key and value are type parameters
    assert isinstance(result.key, TypeParameter)
    assert result.key.name == "K"

    assert isinstance(result.value, TypeParameter)
    assert result.value.name == "V"
