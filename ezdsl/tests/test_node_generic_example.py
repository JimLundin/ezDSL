"""Test real-world example of generic Node with type parameter serialization."""

import pytest
from typing import TypeVar

from ezdsl.schema import extract_type
from ezdsl.types import (
    ListType,
    DictType,
    IntType,
    StrType,
    FloatType,
    NodeType,
    RefType,
    TypeParameter,
)


def test_generic_node_field_extraction():
    """
    Test how a generic Node's fields are extracted.

    Example: class Container[T]:
                 items: list[T]

    When we extract the type of 'items', T should be a TypeParameter,
    not a concrete type.
    """

    # In practice, you'd define it like:
    # class Container[T](Node[list[T]], tag="container"):
    #     items: list[T]

    # Let's manually extract what list[T] would look like
    T = TypeVar("T")

    items_type = extract_type(list[T])

    # The field type is a ListType
    assert isinstance(items_type, ListType)

    # The element is a TypeParameter (the placeholder T), not a concrete type
    assert isinstance(items_type.element, TypeParameter)
    assert items_type.element.name == "T"


def test_complex_generic_node_field():
    """
    Test: class MyNode[T]:
              args: list[dict[str, T]]

    This should serialize as:
    - ListType
      - element: DictType
        - key: StrType
        - value: TypeParameter(name="T")
    """
    T = TypeVar("T")

    # Build the type annotation: list[dict[str, T]]
    result = extract_type(list[dict[str, T]])

    # Outer layer: ListType
    assert isinstance(result, ListType)

    # Middle layer: DictType
    dict_type = result.element
    assert isinstance(dict_type, DictType)

    # dict's first arg: str (concrete type)
    assert isinstance(dict_type.key, StrType)

    # dict's second arg: T (type parameter)
    assert isinstance(dict_type.value, TypeParameter)
    assert dict_type.value.name == "T"
    assert dict_type.value.bound is None


def test_bounded_type_parameter_in_generic_node():
    """
    Test: class NumericNode[T: float]:
              value: T

    The TypeParameter should capture the bound.
    """
    T = TypeVar("T", bound=float)

    result = extract_type(T)

    assert isinstance(result, TypeParameter)
    assert result.name == "T"
    assert result.bound is not None
    assert isinstance(result.bound, FloatType)


def test_type_parameter_vs_concrete_type():
    """
    Demonstrate the difference between:
    1. A type parameter (T in class definition)
    2. A concrete type argument (int when using the class)
    """
    T = TypeVar("T")

    # In the class definition: list[T]
    generic_form = extract_type(list[T])
    assert isinstance(generic_form, ListType)
    assert isinstance(generic_form.element, TypeParameter)  # T is a parameter
    assert generic_form.element.name == "T"

    # When using the class: list[int]
    concrete_form = extract_type(list[int])
    assert isinstance(concrete_form, ListType)
    assert isinstance(concrete_form.element, IntType)  # int is concrete

    # They're both ListType, but their elements are different
    # (TypeParameter vs IntType)
