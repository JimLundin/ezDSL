"""
Workflow/Data Pipeline DSL Example
===================================

This example demonstrates how to build a workflow/data pipeline DSL using nanoDSL.
It covers:

1. Modeling DAG (Directed Acyclic Graph) structures with references
2. Working with shared nodes (diamond patterns)
3. Building execution interpreters for workflows
4. Dependency resolution and topological sorting
5. Serializing and deserializing complex graphs

The workflow DSL supports:
- Data sources (inputs)
- Transformations (map, filter, aggregate)
- Joins (combining multiple data streams)
- Outputs (sinks)
"""

from dataclasses import dataclass
from typing import Literal, Any
from collections import defaultdict

from nanodsl import Node, AST, Ref, Interpreter


# ============================================================================
# Step 1: Define Workflow Nodes
# ============================================================================


class DataSource(Node[list[dict[str, Any]]], tag="wf_source"):
    """
    A data source node (e.g., database, file, API).

    In a real implementation, this would connect to actual data sources.
    For this example, we'll use mock data.
    """

    name: str
    source_type: Literal["database", "file", "api"]


class Transform(Node[list[dict[str, Any]]], tag="wf_transform"):
    """
    Apply a transformation to a data stream.

    The transformation is identified by name (e.g., "uppercase", "filter_active").
    """

    input: Ref[Node[list[dict[str, Any]]]]
    transform_name: str
    params: dict[str, Any] = None


class Filter(Node[list[dict[str, Any]]], tag="wf_filter"):
    """Filter data based on a field condition."""

    input: Ref[Node[list[dict[str, Any]]]]
    field: str
    operator: Literal["==", "!=", ">", "<", ">=", "<=", "contains"]
    value: Any


class Join(Node[list[dict[str, Any]]], tag="wf_join"):
    """
    Join two data streams.

    Supports inner, left, right, and outer joins.
    """

    left: Ref[Node[list[dict[str, Any]]]]
    right: Ref[Node[list[dict[str, Any]]]]
    join_type: Literal["inner", "left", "right", "outer"]
    left_key: str
    right_key: str


class Aggregate(Node[dict[str, Any]], tag="wf_aggregate"):
    """
    Aggregate data (e.g., count, sum, average).
    """

    input: Ref[Node[list[dict[str, Any]]]]
    group_by: list[str]
    aggregations: dict[str, Literal["count", "sum", "avg", "min", "max"]]


class Output(Node[None], tag="wf_output"):
    """
    Output/sink for data (e.g., write to file, database, or console).
    """

    input: Ref[Node[Any]]
    output_type: Literal["file", "database", "console"]
    destination: str


# ============================================================================
# Step 2: Mock Data and Transformations
# ============================================================================


# Mock data for our examples
MOCK_DATA = {
    "users": [
        {"id": 1, "name": "Alice", "age": 30, "active": True},
        {"id": 2, "name": "Bob", "age": 25, "active": True},
        {"id": 3, "name": "Charlie", "age": 35, "active": False},
    ],
    "orders": [
        {"order_id": 101, "user_id": 1, "amount": 50.0},
        {"order_id": 102, "user_id": 2, "amount": 75.0},
        {"order_id": 103, "user_id": 1, "amount": 120.0},
    ],
    "products": [
        {"id": 1, "name": "Widget", "price": 10.0},
        {"id": 2, "name": "Gadget", "price": 20.0},
    ],
}


# Mock transformation functions
def transform_uppercase(data: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    """Transform: uppercase a string field."""
    return [{**row, field: row[field].upper()} for row in data]


def transform_add_field(
    data: list[dict[str, Any]], field: str, value: Any
) -> list[dict[str, Any]]:
    """Transform: add a new field with a constant value."""
    return [{**row, field: value} for row in data]


TRANSFORMS = {
    "uppercase": transform_uppercase,
    "add_field": transform_add_field,
}


# ============================================================================
# Step 3: Execution Context
# ============================================================================


@dataclass
class WorkflowContext:
    """
    Context for workflow execution.

    Attributes:
        data_sources: Mock data sources
        transforms: Available transformation functions
        execution_log: Log of executed nodes (for debugging)
    """

    data_sources: dict[str, list[dict[str, Any]]]
    transforms: dict[str, callable]
    execution_log: list[str]

    def log(self, message: str) -> None:
        """Log an execution step."""
        self.execution_log.append(message)


# ============================================================================
# Step 4: Workflow Interpreter
# ============================================================================


class WorkflowInterpreter(Interpreter[WorkflowContext, Any]):
    """
    Interprets and executes workflow DAGs.

    This interpreter:
    1. Resolves dependencies (follows references)
    2. Executes nodes in dependency order
    3. Caches results to avoid recomputing shared nodes
    4. Logs execution for debugging
    """

    def __init__(self, ast: AST, ctx: WorkflowContext) -> None:
        super().__init__(ast, ctx)
        self._cache: dict[str, Any] = {}

    def eval_ref(self, ref: Ref[Node[Any]]) -> Any:
        """Evaluate a reference with caching."""
        if ref.id not in self._cache:
            self.ctx.log(f"Executing node: {ref.id}")
            node = self.resolve(ref)
            self._cache[ref.id] = self.eval(node)
        else:
            self.ctx.log(f"Using cached result for: {ref.id}")
        return self._cache[ref.id]

    def eval(self, node: Node[Any]) -> Any:
        """Evaluate a workflow node."""
        match node:
            case DataSource(name=name, source_type=src_type):
                if name not in self.ctx.data_sources:
                    raise ValueError(f"Data source '{name}' not found")
                data = self.ctx.data_sources[name]
                self.ctx.log(f"  Loaded {len(data)} rows from {src_type} '{name}'")
                return data

            case Transform(input=input_ref, transform_name=tf_name, params=params):
                input_data = self.eval_ref(input_ref)
                if tf_name not in self.ctx.transforms:
                    raise ValueError(f"Transform '{tf_name}' not found")

                # Apply transformation
                transform_fn = self.ctx.transforms[tf_name]
                if params:
                    result = transform_fn(input_data, **params)
                else:
                    result = transform_fn(input_data)

                self.ctx.log(
                    f"  Applied transform '{tf_name}': {len(input_data)} → {len(result)} rows"
                )
                return result

            case Filter(input=input_ref, field=field, operator=op, value=val):
                input_data = self.eval_ref(input_ref)

                # Apply filter
                filtered = []
                for row in input_data:
                    if field not in row:
                        continue

                    field_val = row[field]
                    match op:
                        case "==":
                            if field_val == val:
                                filtered.append(row)
                        case "!=":
                            if field_val != val:
                                filtered.append(row)
                        case ">":
                            if field_val > val:
                                filtered.append(row)
                        case "<":
                            if field_val < val:
                                filtered.append(row)
                        case ">=":
                            if field_val >= val:
                                filtered.append(row)
                        case "<=":
                            if field_val <= val:
                                filtered.append(row)
                        case "contains":
                            if val in field_val:
                                filtered.append(row)

                self.ctx.log(
                    f"  Filtered on {field} {op} {val}: {len(input_data)} → {len(filtered)} rows"
                )
                return filtered

            case Join(
                left=left_ref,
                right=right_ref,
                join_type=jtype,
                left_key=lkey,
                right_key=rkey,
            ):
                left_data = self.eval_ref(left_ref)
                right_data = self.eval_ref(right_ref)

                # Simple inner join implementation
                if jtype == "inner":
                    # Build index on right side
                    right_index = defaultdict(list)
                    for row in right_data:
                        if rkey in row:
                            right_index[row[rkey]].append(row)

                    # Join
                    result = []
                    for left_row in left_data:
                        if lkey in left_row:
                            key_val = left_row[lkey]
                            if key_val in right_index:
                                for right_row in right_index[key_val]:
                                    # Merge rows
                                    joined = {**left_row, **right_row}
                                    result.append(joined)

                    self.ctx.log(
                        f"  Inner join on {lkey}={rkey}: {len(left_data)} ⨝ {len(right_data)} = {len(result)} rows"
                    )
                    return result
                else:
                    raise NotImplementedError(f"Join type '{jtype}' not implemented")

            case Aggregate(input=input_ref, group_by=group_fields, aggregations=aggs):
                input_data = self.eval_ref(input_ref)

                # Group data
                groups = defaultdict(list)
                for row in input_data:
                    # Create group key
                    key = tuple(row.get(f) for f in group_fields)
                    groups[key].append(row)

                # Compute aggregations
                results = {}
                for group_key, group_rows in groups.items():
                    group_dict = dict(zip(group_fields, group_key))

                    for agg_field, agg_func in aggs.items():
                        match agg_func:
                            case "count":
                                group_dict[f"{agg_field}_count"] = len(group_rows)
                            case "sum":
                                group_dict[f"{agg_field}_sum"] = sum(
                                    row.get(agg_field, 0) for row in group_rows
                                )
                            case "avg":
                                values = [row.get(agg_field, 0) for row in group_rows]
                                group_dict[f"{agg_field}_avg"] = (
                                    sum(values) / len(values) if values else 0
                                )
                            case "min":
                                values = [row.get(agg_field) for row in group_rows]
                                group_dict[f"{agg_field}_min"] = min(values)
                            case "max":
                                values = [row.get(agg_field) for row in group_rows]
                                group_dict[f"{agg_field}_max"] = max(values)

                    results[group_key] = group_dict

                self.ctx.log(
                    f"  Aggregated by {group_fields}: {len(input_data)} → {len(results)} groups"
                )
                return results

            case Output(input=input_ref, output_type=otype, destination=dest):
                input_data = self.eval_ref(input_ref)

                if otype == "console":
                    self.ctx.log(f"  Writing to console: {dest}")
                    print(f"\n{dest}:")
                    print(input_data)
                else:
                    self.ctx.log(f"  Writing to {otype}: {dest}")

                return None

            case _:
                raise NotImplementedError(f"Unknown node type: {type(node)}")


# ============================================================================
# Step 5: Example Workflows
# ============================================================================


def example_simple_pipeline():
    """Simple linear pipeline: source → filter → output"""

    ast = AST(
        root="output",
        nodes={
            "users_src": DataSource(name="users", source_type="database"),
            "active_users": Filter(
                input=Ref(id="users_src"),
                field="active",
                operator="==",
                value=True,
            ),
            "output": Output(
                input=Ref(id="active_users"),
                output_type="console",
                destination="Active Users",
            ),
        },
    )

    ctx = WorkflowContext(
        data_sources=MOCK_DATA, transforms=TRANSFORMS, execution_log=[]
    )

    print("=" * 80)
    print("Example 1: Simple Linear Pipeline")
    print("=" * 80)
    print()
    print("Pipeline: users → filter(active=True) → output")
    print()

    interpreter = WorkflowInterpreter(ast, ctx)
    interpreter.run()

    print()
    print("Execution log:")
    for log_entry in ctx.execution_log:
        print(f"  {log_entry}")
    print()


def example_diamond_pattern():
    """
    Diamond pattern: shared source with multiple paths.

               source
              /      \\
        filter_age  filter_active
              \\      /
                join
    """

    ast = AST(
        root="output",
        nodes={
            "users_src": DataSource(name="users", source_type="database"),
            # Left branch: filter by age
            "age_filter": Filter(
                input=Ref(id="users_src"), field="age", operator=">=", value=30
            ),
            # Right branch: filter by active status
            "active_filter": Filter(
                input=Ref(id="users_src"), field="active", operator="==", value=True
            ),
            # Join the two filtered streams
            "joined": Join(
                left=Ref(id="age_filter"),
                right=Ref(id="active_filter"),
                join_type="inner",
                left_key="id",
                right_key="id",
            ),
            "output": Output(
                input=Ref(id="joined"),
                output_type="console",
                destination="Users: age>=30 AND active=True",
            ),
        },
    )

    ctx = WorkflowContext(
        data_sources=MOCK_DATA, transforms=TRANSFORMS, execution_log=[]
    )

    print("=" * 80)
    print("Example 2: Diamond Pattern (Shared Source)")
    print("=" * 80)
    print()
    print("Pipeline:")
    print("              users_src")
    print("             /         \\")
    print("       age>=30        active=True")
    print("             \\         /")
    print("              join(id)")
    print()

    interpreter = WorkflowInterpreter(ast, ctx)
    interpreter.run()

    print()
    print("Execution log:")
    for log_entry in ctx.execution_log:
        print(f"  {log_entry}")
    print()
    print("Note: 'users_src' is only executed once (cached for both branches)")
    print()


def example_complex_dag():
    """
    Complex DAG with multiple sources, joins, and transformations.
    """

    ast = AST(
        root="output",
        nodes={
            # Data sources
            "users_src": DataSource(name="users", source_type="database"),
            "orders_src": DataSource(name="orders", source_type="database"),
            # Filter active users
            "active_users": Filter(
                input=Ref(id="users_src"), field="active", operator="==", value=True
            ),
            # Join users with orders
            "user_orders": Join(
                left=Ref(id="active_users"),
                right=Ref(id="orders_src"),
                join_type="inner",
                left_key="id",
                right_key="user_id",
            ),
            # Filter high-value orders
            "high_value": Filter(
                input=Ref(id="user_orders"),
                field="amount",
                operator=">",
                value=60.0,
            ),
            # Aggregate by user
            "aggregated": Aggregate(
                input=Ref(id="high_value"),
                group_by=["name"],
                aggregations={"amount": "sum", "order_id": "count"},
            ),
            "output": Output(
                input=Ref(id="aggregated"),
                output_type="console",
                destination="High-Value Orders Summary",
            ),
        },
    )

    ctx = WorkflowContext(
        data_sources=MOCK_DATA, transforms=TRANSFORMS, execution_log=[]
    )

    print("=" * 80)
    print("Example 3: Complex DAG")
    print("=" * 80)
    print()
    print("Pipeline:")
    print("  users → filter(active) \\")
    print("                          join(id=user_id) → filter(amount>60) → aggregate → output")
    print("  orders ________________/")
    print()

    interpreter = WorkflowInterpreter(ast, ctx)
    interpreter.run()

    print()
    print("Execution log:")
    for log_entry in ctx.execution_log:
        print(f"  {log_entry}")
    print()


def example_serialization():
    """Demonstrate serialization of a workflow DAG."""

    # Build workflow
    ast = AST(
        root="output",
        nodes={
            "src": DataSource(name="users", source_type="database"),
            "filtered": Filter(
                input=Ref(id="src"), field="age", operator=">=", value=25
            ),
            "output": Output(
                input=Ref(id="filtered"),
                output_type="console",
                destination="Filtered Users",
            ),
        },
    )

    print("=" * 80)
    print("Example 4: Workflow Serialization")
    print("=" * 80)
    print()

    # Serialize to JSON
    json_str = ast.to_json()
    print("Serialized workflow:")
    print(json_str)
    print()

    # Deserialize
    restored_ast = AST.from_json(json_str)
    print(f"Successfully deserialized: {restored_ast == ast}")
    print()

    # Execute restored workflow
    ctx = WorkflowContext(
        data_sources=MOCK_DATA, transforms=TRANSFORMS, execution_log=[]
    )
    interpreter = WorkflowInterpreter(restored_ast, ctx)
    interpreter.run()

    print()


# ============================================================================
# Main: Run All Examples
# ============================================================================


def main():
    """Run all workflow DSL examples."""

    example_simple_pipeline()
    example_diamond_pattern()
    example_complex_dag()
    example_serialization()

    print("All examples completed! ✓")


if __name__ == "__main__":
    main()
