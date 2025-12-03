"""
ezdsl - Easy Domain Specific Languages

A minimal AST node type system for Python 3.12+
"""

from nanodsl.nodes import (
    # Core types
    Node,
    Ref,
    NodeRef,
    Child,
)

from nanodsl.types import (
    # Type definitions
    TypeDef,
    IntType,
    FloatType,
    StrType,
    BoolType,
    NoneType,
    ListType,
    DictType,
    SetType,
    TupleType,
    LiteralType,
    NodeType,
    RefType,
    UnionType,
    TypeParameter,
    TypeParameterRef,
    ExternalType,
    CustomType,
    ExternalTypeRecord,
    CustomTypeRecord,
)

from nanodsl.serialization import (
    # Serialization
    to_dict,
    from_dict,
    to_json,
    from_json,
)

from nanodsl.schema import (
    # Schema extraction
    extract_type,
    node_schema,
    all_schemas,
)

from nanodsl.ast import (
    AST,
)

__all__ = [
    # Core types
    "Node",
    "Ref",
    "NodeRef",
    "Child",
    "AST",
    # Type definitions
    "TypeDef",
    "IntType",
    "FloatType",
    "StrType",
    "BoolType",
    "NoneType",
    "ListType",
    "DictType",
    "SetType",
    "TupleType",
    "LiteralType",
    "NodeType",
    "RefType",
    "UnionType",
    "TypeParameter",
    "TypeParameterRef",
    "ExternalType",
    "CustomType",
    "ExternalTypeRecord",
    "CustomTypeRecord",
    # Serialization
    "to_dict",
    "from_dict",
    "to_json",
    "from_json",
    # Schema extraction
    "extract_type",
    "node_schema",
    "all_schemas",
]
