"""
ezdsl - Easy Domain Specific Languages

A minimal AST node type system for Python 3.12+
"""

from ezdsl.core import (
    # Core types
    Node,
    Ref,
    NodeRef,
    Child,
    AST,

    # Type definitions
    TypeDef,
    PrimitiveType,
    NodeType,
    RefType,
    UnionType,
    GenericType,
    TypeVarType,

    # Serialization
    to_dict,
    from_dict,
    to_json,
    from_json,

    # Schema extraction
    extract_type,
    node_schema,
    all_schemas,

    # Constants
    PRIMITIVES,
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
    "PrimitiveType",
    "NodeType",
    "RefType",
    "UnionType",
    "GenericType",
    "TypeVarType",

    # Serialization
    "to_dict",
    "from_dict",
    "to_json",
    "from_json",

    # Schema extraction
    "extract_type",
    "node_schema",
    "all_schemas",

    # Constants
    "PRIMITIVES",
]
