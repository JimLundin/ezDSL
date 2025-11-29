"""
ezdsl - Easy Domain Specific Languages

A minimal AST node type system for Python 3.12+
"""

from ezdsl.nodes import (
    # Core types
    Node,
    Ref,
    NodeRef,
    Child,
)

from ezdsl.types import (
    # Type definitions
    TypeDef,
    IntType,
    FloatType,
    StrType,
    BoolType,
    NoneType,
    ListType,
    DictType,
    NodeType,
    RefType,
    UnionType,
    TypeParameter,
)

from ezdsl.serialization import (
    # Serialization
    to_dict,
    from_dict,
    to_json,
    from_json,
)

from ezdsl.schema import (
    # Schema extraction
    extract_type,
    node_schema,
    all_schemas,
)

from ezdsl.ast import (
    AST,
)

__all__ = [
    # Core types
    "Node",
    "Ref",
    "NodeRef",
    "Child",
    "AST",

    # Type definitions (TypeDef has classmethod register() and get_registered_type())
    "TypeDef",
    "IntType",
    "FloatType",
    "StrType",
    "BoolType",
    "NoneType",
    "ListType",
    "DictType",
    "NodeType",
    "RefType",
    "UnionType",
    "TypeParameter",

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
