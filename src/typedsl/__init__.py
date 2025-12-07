"""typeDSL - Type-safe AST node system for Python 3.12+."""

from typedsl.adapters import (
    # Format adapters
    FormatAdapter,
    JSONAdapter,
)
from typedsl.ast import (
    AST,
    Interpreter,
)
from typedsl.nodes import (
    Child,
    # Core types
    Node,
    NodeRef,
    Ref,
)
from typedsl.schema import (
    FieldSchema,
    # Schema dataclasses
    NodeSchema,
    all_schemas,
    # Schema extraction
    extract_type,
    node_schema,
)
from typedsl.serialization import (
    from_dict,
    from_json,
    # Serialization
    to_dict,
    to_json,
)
from typedsl.types import (
    BoolType,
    DictType,
    ExternalType,
    ExternalTypeRecord,
    FloatType,
    IntType,
    ListType,
    LiteralType,
    NodeType,
    NoneType,
    RefType,
    SetType,
    StrType,
    TupleType,
    # Type definitions
    TypeDef,
    TypeParameter,
    TypeParameterRef,
    UnionType,
)

__all__ = [
    "AST",
    "BoolType",
    "Child",
    "DictType",
    "ExternalType",
    "ExternalTypeRecord",
    "FieldSchema",
    "FloatType",
    # Format adapters
    "FormatAdapter",
    "IntType",
    "Interpreter",
    "JSONAdapter",
    "ListType",
    "LiteralType",
    # Core types
    "Node",
    "NodeRef",
    "NodeSchema",
    "NodeType",
    "NoneType",
    "Ref",
    "RefType",
    "SetType",
    "StrType",
    "TupleType",
    # Type definitions
    "TypeDef",
    "TypeParameter",
    "TypeParameterRef",
    "UnionType",
    "all_schemas",
    # Schema extraction
    "extract_type",
    "from_dict",
    "from_json",
    "node_schema",
    # Serialization
    "to_dict",
    "to_json",
]
