"""nanoDSL - Type-safe AST node system for Python 3.12+"""

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
    ExternalTypeRecord,
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
    # Schema dataclasses
    NodeSchema,
    FieldSchema,
)

from nanodsl.adapters import (
    # Format adapters
    FormatAdapter,
    JSONAdapter,
)

from nanodsl.ast import (
    AST,
    Interpreter,
)

__all__ = [
    # Core types
    "Node",
    "Ref",
    "NodeRef",
    "Child",
    "AST",
    "Interpreter",
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
    "ExternalTypeRecord",
    # Serialization
    "to_dict",
    "from_dict",
    "to_json",
    "from_json",
    # Schema extraction
    "extract_type",
    "node_schema",
    "all_schemas",
    "NodeSchema",
    "FieldSchema",
    # Format adapters
    "FormatAdapter",
    "JSONAdapter",
]
