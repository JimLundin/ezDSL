"""Tests for user-defined custom types in the DSL."""

from nanodsl.nodes import Node
from nanodsl.types import (
    TypeDef,
    IntType,
    StrType,
    NodeType,
)
from nanodsl.schema import extract_type, node_schema
from nanodsl.serialization import to_dict, from_dict


# =============================================================================
# Define Custom Types
# =============================================================================


# User defines marker classes and registers them with TypeDef.register()
@TypeDef.register
class DataFrame:
    """User-defined DataFrame type marker."""


@TypeDef.register
class Matrix:
    """User-defined Matrix type marker."""


# Get the TypeDef classes for assertions
DataFrameType = TypeDef.get_registered_type(DataFrame)
MatrixType = TypeDef.get_registered_type(Matrix)


# =============================================================================
# Define DSL Nodes using custom types
# =============================================================================


class FetchData(Node[DataFrame], tag="fetch_data"):
    """Fetches data and returns a DataFrame."""

    query: str


class FilterData(Node[DataFrame], tag="filter_data"):
    """Filters a DataFrame."""

    source: Node[DataFrame]
    condition: str


class ToMatrix(Node[Matrix], tag="to_matrix"):
    """Converts a DataFrame to a Matrix."""

    df: Node[DataFrame]


class MatrixMultiply(Node[Matrix], tag="matrix_multiply"):
    """Multiplies two matrices."""

    left: Node[Matrix]
    right: Node[Matrix]


# =============================================================================
# Tests
# =============================================================================


class TestCustomTypeRegistration:
    """Test custom type registration."""

    def test_register_custom_type(self):
        """Test registering a custom type."""

        class MyType:
            pass

        # Use TypeDef.register decorator
        TypeDef.register(MyType)
        result = TypeDef.get_registered_type(MyType)
        assert result is not None
        # New implementation returns CustomType instance with tag field
        from nanodsl.types import CustomType

        assert isinstance(result, CustomType)
        assert result.tag == "mytype"

    def test_get_unregistered_type(self):
        """Test getting an unregistered type returns None."""

        class UnregisteredType:
            pass

        assert TypeDef.get_registered_type(UnregisteredType) is None


class TestCustomTypeExtraction:
    """Test extracting custom types."""

    def test_extract_dataframe_type(self):
        """Test extracting DataFrame type."""
        result = extract_type(DataFrame)
        assert isinstance(result, DataFrameType)
        assert result._tag == "dataframe"

    def test_extract_matrix_type(self):
        """Test extracting Matrix type."""
        result = extract_type(Matrix)
        assert isinstance(result, MatrixType)
        assert result._tag == "matrix"

    def test_extract_node_with_custom_return_type(self):
        """Test extracting a Node with custom return type."""
        from typing import get_type_hints

        hints = get_type_hints(FetchData)

        # The query field is str
        query_type = extract_type(hints["query"])
        assert isinstance(query_type, StrType)

    def test_extract_node_field_with_custom_type(self):
        """Test extracting a field that uses custom type in Node."""
        from typing import get_type_hints

        hints = get_type_hints(FilterData)

        # The source field is Node[DataFrame]
        source_type = extract_type(hints["source"])
        assert isinstance(source_type, NodeType)
        assert isinstance(source_type.returns, DataFrameType)


class TestCustomTypeNodeSchema:
    """Test node schemas with custom types."""

    def test_fetch_data_schema(self):
        """Test schema for FetchData node."""
        schema = node_schema(FetchData)

        assert schema["tag"] == "fetch_data"
        assert schema["returns"] == {"tag": "dataframe"}
        assert len(schema["fields"]) == 1
        assert schema["fields"][0]["name"] == "query"
        assert schema["fields"][0]["type"] == {"tag": "str"}

    def test_filter_data_schema(self):
        """Test schema for FilterData node with custom type field."""
        schema = node_schema(FilterData)

        assert schema["tag"] == "filter_data"
        assert schema["returns"] == {"tag": "dataframe"}

        # Find the source field
        source_field = next(f for f in schema["fields"] if f["name"] == "source")
        assert source_field["type"] == {
            "tag": "node",
            "returns": {"tag": "dataframe"},
        }

    def test_to_matrix_schema(self):
        """Test schema for node that converts between custom types."""
        schema = node_schema(ToMatrix)

        assert schema["tag"] == "to_matrix"
        assert schema["returns"] == {"tag": "matrix"}

        # Find the df field
        df_field = next(f for f in schema["fields"] if f["name"] == "df")
        assert df_field["type"] == {
            "tag": "node",
            "returns": {"tag": "dataframe"},
        }


class TestCustomTypeSerialization:
    """Test serialization of custom types."""

    def test_serialize_dataframe_type(self):
        """Test serializing DataFrameType."""
        df_type = DataFrameType()
        result = to_dict(df_type)
        assert result == {"tag": "dataframe"}

    def test_deserialize_dataframe_type(self):
        """Test deserializing DataFrameType."""
        data = {"tag": "dataframe"}
        result = from_dict(data)
        assert isinstance(result, DataFrameType)

    def test_serialize_matrix_type(self):
        """Test serializing MatrixType."""
        matrix_type = MatrixType()
        result = to_dict(matrix_type)
        assert result == {"tag": "matrix"}

    def test_deserialize_matrix_type(self):
        """Test deserializing MatrixType."""
        data = {"tag": "matrix"}
        result = from_dict(data)
        assert isinstance(result, MatrixType)


class TestIDESupport:
    """
    Test that IDE support works correctly.

    These tests verify that the type system works with Python's type system
    so IDEs can provide proper type checking and autocomplete.
    """

    def test_node_with_custom_type_is_valid(self):
        """Test that Node[DataFrame] is valid."""
        # This compiles without type errors
        fetch: Node[DataFrame] = FetchData(query="SELECT * FROM users")
        assert isinstance(fetch, FetchData)

    def test_node_field_with_custom_type_is_valid(self):
        """Test that fields with custom types in Nodes work."""
        fetch = FetchData(query="SELECT * FROM users")
        filter_node: Node[DataFrame] = FilterData(source=fetch, condition="age > 18")
        assert isinstance(filter_node, FilterData)

    def test_type_conversion_nodes(self):
        """Test nodes that convert between custom types."""
        fetch = FetchData(query="SELECT * FROM data")
        matrix: Node[Matrix] = ToMatrix(df=fetch)
        assert isinstance(matrix, ToMatrix)


class TestComplexCustomTypeExamples:
    """Test more complex examples with custom types."""

    def test_nested_operations(self):
        """Test chaining operations with custom types."""
        # Fetch -> Filter -> ToMatrix
        fetch = FetchData(query="SELECT * FROM sales")
        filtered = FilterData(source=fetch, condition="revenue > 1000")
        matrix = ToMatrix(df=filtered)

        schema = node_schema(ToMatrix)
        assert schema["returns"] == {"tag": "matrix"}

    def test_matrix_operations(self):
        """Test operations on Matrix types."""
        fetch1 = FetchData(query="SELECT * FROM data1")
        fetch2 = FetchData(query="SELECT * FROM data2")

        matrix1 = ToMatrix(df=fetch1)
        matrix2 = ToMatrix(df=fetch2)

        result: Node[Matrix] = MatrixMultiply(left=matrix1, right=matrix2)

        schema = node_schema(MatrixMultiply)
        assert schema["returns"] == {"tag": "matrix"}
        assert len(schema["fields"]) == 2
