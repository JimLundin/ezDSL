"""Format adapters for serialization."""

from __future__ import annotations

from abc import ABC, abstractmethod
import dataclasses
from typing import Any

from nanodsl.nodes import Node, Ref
from nanodsl.types import TypeDef
from nanodsl.schema import NodeSchema, FieldSchema


class FormatAdapter(ABC):
    """Base class for format-specific serialization."""

    @abstractmethod
    def serialize_node(self, node: Node) -> Any: ...

    @abstractmethod
    def deserialize_node(self, data: Any) -> Node: ...

    @abstractmethod
    def serialize_typedef(self, typedef: TypeDef) -> Any: ...

    @abstractmethod
    def deserialize_typedef(self, data: Any) -> TypeDef: ...

    @abstractmethod
    def serialize_node_schema(self, schema: NodeSchema) -> Any: ...


class JSONAdapter(FormatAdapter):
    """JSON serialization adapter."""

    def serialize_node(self, node: Node) -> dict:
        result = {"tag": type(node)._tag}
        for field in dataclasses.fields(node):
            if not field.name.startswith("_"):
                result[field.name] = self._serialize_value(getattr(node, field.name))
        return result

    def deserialize_node(self, data: dict) -> Node:
        tag = data["tag"]
        if tag == "ref":
            return Ref(id=data["id"])

        node_cls = Node._registry.get(tag)
        if node_cls is None:
            raise ValueError(f"Unknown node tag: {tag}")

        field_values = {}
        for field in dataclasses.fields(node_cls):
            if not field.name.startswith("_") and field.name in data:
                field_values[field.name] = self._deserialize_value(data[field.name])
        return node_cls(**field_values)

    def deserialize_typedef(self, data: dict) -> TypeDef:
        tag = data["tag"]
        typedef_cls = TypeDef._registry.get(tag)
        if typedef_cls is None:
            raise ValueError(f"Unknown TypeDef tag: {tag}")

        field_values = {}
        for field in dataclasses.fields(typedef_cls):
            if not field.name.startswith("_") and field.name in data:
                field_values[field.name] = self._deserialize_value(data[field.name])
        return typedef_cls(**field_values)

    def serialize_typedef(self, typedef: TypeDef) -> dict:
        result = {"tag": type(typedef)._tag}
        for field in dataclasses.fields(typedef):
            if not field.name.startswith("_"):
                result[field.name] = self._serialize_value(getattr(typedef, field.name))
        return result

    def serialize_node_schema(self, schema: NodeSchema) -> dict:
        return {
            "tag": schema.tag,
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
        if isinstance(value, (list, tuple)):
            return [self._serialize_value(item) for item in value]
        if isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        return value

    def _deserialize_value(self, value: Any) -> Any:
        if isinstance(value, dict) and "tag" in value:
            tag = value["tag"]
            if tag == "ref":
                return Ref(id=value["id"])
            if tag in Node._registry:
                return self.deserialize_node(value)
            if tag in TypeDef._registry:
                return self.deserialize_typedef(value)
            raise ValueError(f"Unknown tag: {tag}")
        if isinstance(value, list):
            return [self._deserialize_value(item) for item in value]
        if isinstance(value, dict):
            return {k: self._deserialize_value(v) for k, v in value.items()}
        return value
