"""Format adapters for serialization."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import fields
from typing import TYPE_CHECKING, Any, TypedDict

from typedsl.nodes import Node, Ref
from typedsl.types import TypeDef

if TYPE_CHECKING:
    from typedsl.schema import NodeSchema


class SerializedFieldSchema(TypedDict):
    """Serialized field schema structure."""

    name: str
    type: dict[str, Any]


class SerializedNodeSchema(TypedDict):
    """Serialized node schema structure."""

    tag: str
    signature: dict[str, Any]  # Signature kwargs
    type_params: list[dict[str, Any]]
    returns: dict[str, Any]
    fields: list[SerializedFieldSchema]


class FormatAdapter(ABC):
    """Base class for format-specific serialization."""

    @abstractmethod
    def serialize_node(self, node: Node[Any]) -> dict[str, Any]:
        """Serialize a Node to dictionary format."""
        ...

    @abstractmethod
    def deserialize_node(self, data: dict[str, Any]) -> Node[Any]:
        """Deserialize a dictionary to a Node."""
        ...

    @abstractmethod
    def serialize_typedef(self, typedef: TypeDef) -> dict[str, Any]:
        """Serialize a TypeDef to dictionary format."""
        ...

    @abstractmethod
    def deserialize_typedef(self, data: dict[str, Any]) -> TypeDef:
        """Deserialize a dictionary to a TypeDef."""
        ...

    @abstractmethod
    def serialize_node_schema(self, schema: NodeSchema) -> SerializedNodeSchema:
        """Serialize a NodeSchema to dictionary format."""
        ...


class JSONAdapter(FormatAdapter):
    """JSON serialization adapter."""

    def serialize_node(self, node: Node[Any]) -> dict[str, Any]:
        """Serialize a Node to a JSON-compatible dictionary."""
        result = {
            field.name: self._serialize_value(getattr(node, field.name))
            for field in fields(node)
            if not field.name.startswith("_")
        }
        result["tag"] = type(node).tag
        return result

    def deserialize_node(self, data: dict[str, Any]) -> Node[Any]:
        """Deserialize a JSON-compatible dictionary to a Node."""
        tag = data["tag"]
        node_cls = Node.registry.get(tag)
        if node_cls is None:
            msg = f"Unknown node tag: {tag}"
            raise ValueError(msg)

        field_values = {
            field.name: self._deserialize_value(data[field.name])
            for field in fields(node_cls)
            if not field.name.startswith("_") and field.name in data
        }
        return node_cls(**field_values)

    def deserialize_typedef(self, data: dict[str, Any]) -> TypeDef:
        """Deserialize a JSON-compatible dictionary to a TypeDef."""
        tag = data["tag"]
        typedef_cls = TypeDef.registry.get(tag)
        if typedef_cls is None:
            msg = f"Unknown TypeDef tag: {tag}"
            raise ValueError(msg)

        field_values = {
            field.name: self._deserialize_value(data[field.name])
            for field in fields(typedef_cls)
            if not field.name.startswith("_") and field.name in data
        }
        return typedef_cls(**field_values)

    def serialize_typedef(self, typedef: TypeDef) -> dict[str, Any]:
        """Serialize a TypeDef to a JSON-compatible dictionary."""
        result = {
            field.name: self._serialize_value(getattr(typedef, field.name))
            for field in fields(typedef)
            if not field.name.startswith("_")
        }
        result["tag"] = type(typedef).tag
        return result

    def serialize_node_schema(self, schema: NodeSchema) -> SerializedNodeSchema:
        """Serialize a NodeSchema to a JSON-compatible dictionary."""
        return {
            "tag": schema.tag,
            "signature": schema.signature,
            "type_params": [self.serialize_typedef(tp) for tp in schema.type_params],
            "returns": self.serialize_typedef(schema.returns),
            "fields": [
                {"name": f.name, "type": self.serialize_typedef(f.type)}
                for f in schema.fields
            ],
        }

    def _serialize_value(self, value: Any) -> Any:
        if isinstance(value, Node):
            return self.serialize_node(value)
        if isinstance(value, Ref):
            return {"tag": "ref", "id": value.id}
        if isinstance(value, TypeDef):
            return self.serialize_typedef(value)
        if isinstance(value, list | tuple):
            return [self._serialize_value(item) for item in value]
        if isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        return value

    def _deserialize_value(self, value: Any) -> Any:
        if isinstance(value, dict) and "tag" in value:
            tag = value["tag"]
            if tag == "ref":
                return Ref(id=value["id"])
            if tag in Node.registry:
                return self.deserialize_node(value)
            if tag in TypeDef.registry:
                return self.deserialize_typedef(value)
            msg = f"Unknown tag: {tag}"
            raise ValueError(msg)
        if isinstance(value, list):
            return [self._deserialize_value(item) for item in value]
        if isinstance(value, dict):
            return {k: self._deserialize_value(v) for k, v in value.items()}
        return value
