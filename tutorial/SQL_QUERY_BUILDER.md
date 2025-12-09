# Tutorial: Building a Type-Safe SQL Query Builder

Build a SQL query builder that constructs type-safe queries as ASTs, validates them, and generates SQL.

**Time**: 45 minutes
**Difficulty**: Intermediate
**Prerequisites**: Complete `TUTORIAL.md` first

---

## What We're Building

A SQL query builder that:
- Constructs queries as type-safe ASTs
- Validates queries against schema
- Generates correct SQL strings
- Prevents SQL injection
- Catches type mismatches at construction time

**Example usage:**
```python
query = Select(
    columns=["name", "email"],
    table="users",
    where=Greater(column="age", value=25)
)
```

**Generates:** `SELECT name, email FROM users WHERE age > 25`

---

## Step 1: Define SQL Nodes (12 min)

### Table and Column References

```python
from typedsl import Node, Ref, AST, Interpreter
from typing import Literal, Any

class Table(Node[str], tag="sql_table"):
    """Reference to a database table."""
    name: str

class Column[T](Node[T], tag="sql_column"):
    """
    Type-safe column reference.

    Type parameter T is the column's data type.
    """
    table: str | None  # Optional table qualifier
    name: str
```

**Why generic Column?**
```python
# Type-safe columns
age_col = Column[int](table="users", name="age")
name_col = Column[str](table="users", name="name")

# Type system knows:
# - age_col produces int
# - name_col produces str
```

### Literal Values

```python
class Literal[T](Node[T], tag="sql_literal"):
    """
    Type-safe literal value.

    Type parameter T is the value's type.
    """
    value: T

# Examples
int_lit = Literal[int](value=25)
str_lit = Literal[str](value="John")
```

### Comparison Operators

```python
class Comparison[T](Node[bool], tag="sql_compare"):
    """
    Type-safe comparison: left op right

    Type parameter T ensures both sides have the same type.
    """
    op: Literal["=", "!=", ">", "<", ">=", "<=", "LIKE"]
    left: Ref[Node[T]]    # Left operand
    right: Ref[Node[T]]   # Right operand must match type

class LogicalOp(Node[bool], tag="sql_logical"):
    """Combine boolean expressions with AND/OR."""
    op: Literal["AND", "OR"]
    left: Ref[Node[bool]]
    right: Ref[Node[bool]]
```

**Type safety in action:**
```python
# ✓ Type-safe: comparing int to int
age_check = Comparison[int](
    op=">=",
    left=Ref(id="age_column"),   # Column[int]
    right=Ref(id="age_value")    # Literal[int]
)

# ✗ Type error: comparing int to str
broken = Comparison[???](  # What type? Can't be both!
    op="=",
    left=Ref(id="age_column"),   # int
    right=Ref(id="name_value")   # str
)
```

### Query Types

```python
class Select(Node[list[dict]], tag="sql_select"):
    """SELECT query."""
    columns: list[str]                  # Column names to select
    table: str                          # Table name
    where: Ref[Node[bool]] | None = None  # Optional WHERE clause
    limit: int | None = None            # Optional LIMIT

class Insert(Node[None], tag="sql_insert"):
    """INSERT query."""
    table: str                          # Table name
    columns: list[str]                  # Column names
    values: list[Ref[Node[Any]]]        # Values to insert

class Update(Node[None], tag="sql_update"):
    """UPDATE query."""
    table: str                          # Table name
    set_clause: dict[str, Ref[Node[Any]]]  # column -> value
    where: Ref[Node[bool]] | None = None   # Optional WHERE
```

---

## Step 2: SQL Generator (10 min)

### Concept: Generate SQL Strings from AST

```python
class SQLGenerator(Interpreter[None, str]):
    """Convert AST to SQL string."""

    def eval(self, node: Node[Any]) -> str:
        match node:
            # Literals
            case Literal(value=v):
                # Properly escape values
                if v is None:
                    return "NULL"
                elif isinstance(v, str):
                    # Escape single quotes
                    escaped = v.replace("'", "''")
                    return f"'{escaped}'"
                elif isinstance(v, bool):
                    return "TRUE" if v else "FALSE"
                else:
                    return str(v)

            # Columns
            case Column(table=t, name=n):
                if t:
                    return f"{t}.{n}"
                return n

            # Comparisons
            case Comparison(op=op, left=l, right=r):
                left_sql = self.eval(self.resolve(l))
                right_sql = self.eval(self.resolve(r))
                return f"{left_sql} {op} {right_sql}"

            # Logical operators
            case LogicalOp(op=op, left=l, right=r):
                left_sql = self.eval(self.resolve(l))
                right_sql = self.eval(self.resolve(r))
                return f"({left_sql} {op} {right_sql})"

            # SELECT
            case Select(columns=cols, table=t, where=w, limit=lim):
                sql = f"SELECT {', '.join(cols)} FROM {t}"

                if w:
                    where_sql = self.eval(self.resolve(w))
                    sql += f" WHERE {where_sql}"

                if lim:
                    sql += f" LIMIT {lim}"

                return sql

            # INSERT
            case Insert(table=t, columns=cols, values=vals):
                val_strs = [self.eval(self.resolve(v)) for v in vals]
                return (
                    f"INSERT INTO {t} ({', '.join(cols)}) "
                    f"VALUES ({', '.join(val_strs)})"
                )

            # UPDATE
            case Update(table=t, set_clause=sets, where=w):
                set_strs = [
                    f"{col} = {self.eval(self.resolve(val))}"
                    for col, val in sets.items()
                ]
                sql = f"UPDATE {t} SET {', '.join(set_strs)}"

                if w:
                    where_sql = self.eval(self.resolve(w))
                    sql += f" WHERE {where_sql}"

                return sql

            case _:
                raise NotImplementedError(f"Unknown node: {type(node)}")
```

### Test It

```python
# Build query: SELECT name, email FROM users WHERE age >= 25 LIMIT 10
ast = AST(
    root="query",
    nodes={
        # WHERE clause: age >= 25
        "age_col": Column[int](table=None, name="age"),
        "age_val": Literal[int](value=25),
        "age_check": Comparison[int](
            op=">=",
            left=Ref(id="age_col"),
            right=Ref(id="age_val")
        ),

        # SELECT query
        "query": Select(
            columns=["name", "email"],
            table="users",
            where=Ref(id="age_check"),
            limit=10
        )
    }
)

# Generate SQL
generator = SQLGenerator(ast, None)
sql = generator.run()
print(sql)
# Output: SELECT name, email FROM users WHERE age >= 25 LIMIT 10
```

### SQL Injection Prevention

The AST approach prevents injection:

```python
# ✓ Safe: value is properly escaped
user_input = "Robert'; DROP TABLE users; --"
ast = AST(
    root="query",
    nodes={
        "name_col": Column[str](table=None, name="name"),
        "name_val": Literal[str](value=user_input),  # Will be escaped!
        "name_check": Comparison[str](
            op="=",
            left=Ref(id="name_col"),
            right=Ref(id="name_val")
        ),
        "query": Select(
            columns=["*"],
            table="users",
            where=Ref(id="name_check")
        )
    }
)

sql = SQLGenerator(ast, None).run()
print(sql)
# Output: SELECT * FROM users WHERE name = 'Robert''; DROP TABLE users; --'
# Note: Single quote is escaped to ''
```

---

## Step 3: Schema Validation (12 min)

### Define Database Schema

```python
from dataclasses import dataclass

@dataclass
class Schema:
    """Database schema: tables and their columns."""
    tables: dict[str, dict[str, type]]  # table -> {column: type}

# Example schema
db_schema = Schema(tables={
    "users": {
        "id": int,
        "name": str,
        "email": str,
        "age": int,
        "active": bool
    },
    "orders": {
        "id": int,
        "user_id": int,
        "amount": float,
        "created_at": str
    }
})
```

### Validator Interpreter

```python
class QueryValidator(Interpreter[Schema, None]):
    """Validate queries against schema."""

    def eval(self, node: Node[Any]) -> None:
        match node:
            # Validate SELECT
            case Select(columns=cols, table=t, where=w, limit=lim):
                # Check table exists
                if t not in self.ctx.tables:
                    raise ValueError(f"Unknown table: {t}")

                # Check columns exist
                table_schema = self.ctx.tables[t]
                for col in cols:
                    if col != "*" and col not in table_schema:
                        raise ValueError(f"Unknown column: {t}.{col}")

                # Validate WHERE clause
                if w:
                    self.eval(self.resolve(w))

                # Validate limit is positive
                if lim is not None and lim <= 0:
                    raise ValueError(f"LIMIT must be positive: {lim}")

            # Validate INSERT
            case Insert(table=t, columns=cols, values=vals):
                if t not in self.ctx.tables:
                    raise ValueError(f"Unknown table: {t}")

                table_schema = self.ctx.tables[t]

                # Check all columns exist
                for col in cols:
                    if col not in table_schema:
                        raise ValueError(f"Unknown column: {t}.{col}")

                # Check value count matches column count
                if len(cols) != len(vals):
                    raise ValueError(
                        f"Column count ({len(cols)}) doesn't match value count ({len(vals)})"
                    )

                # Validate each value
                for val_ref in vals:
                    self.eval(self.resolve(val_ref))

            # Validate UPDATE
            case Update(table=t, set_clause=sets, where=w):
                if t not in self.ctx.tables:
                    raise ValueError(f"Unknown table: {t}")

                table_schema = self.ctx.tables[t]

                # Check all columns exist
                for col in sets.keys():
                    if col not in table_schema:
                        raise ValueError(f"Unknown column: {t}.{col}")

                # Validate values
                for val_ref in sets.values():
                    self.eval(self.resolve(val_ref))

                # Validate WHERE clause
                if w:
                    self.eval(self.resolve(w))

            # Validate comparisons (could add type checking)
            case Comparison(left=l, right=r):
                self.eval(self.resolve(l))
                self.eval(self.resolve(r))

            # Validate logical ops
            case LogicalOp(left=l, right=r):
                self.eval(self.resolve(l))
                self.eval(self.resolve(r))

            # Columns and literals are always valid
            case Column() | Literal():
                pass

            case _:
                raise NotImplementedError(f"Unknown node: {type(node)}")
```

### Use Validation

```python
# Valid query
ast = AST(
    root="query",
    nodes={
        "age_col": Column[int](table=None, name="age"),
        "age_val": Literal[int](value=25),
        "age_check": Comparison[int](
            op=">=",
            left=Ref(id="age_col"),
            right=Ref(id="age_val")
        ),
        "query": Select(
            columns=["name", "email"],
            table="users",
            where=Ref(id="age_check")
        )
    }
)

# Validate before generating SQL
validator = QueryValidator(ast, db_schema)
try:
    validator.run()
    print("✓ Query is valid")

    # Generate SQL
    sql = SQLGenerator(ast, None).run()
    print(f"SQL: {sql}")
except ValueError as e:
    print(f"✗ Validation error: {e}")
```

### Example: Catch Invalid Queries

```python
# Invalid: unknown column
ast = AST(
    root="query",
    nodes={
        "query": Select(
            columns=["name", "phone"],  # 'phone' doesn't exist!
            table="users"
        )
    }
)

validator = QueryValidator(ast, db_schema)
try:
    validator.run()
except ValueError as e:
    print(f"Caught error: {e}")
    # Output: Caught error: Unknown column: users.phone
```

---

## Step 4: Type-Safe Aggregations (8 min)

### Generic Aggregate Node

```python
class Aggregate[T, R](Node[R], tag="sql_aggregate"):
    """
    Generic aggregation.

    Type parameters:
    - T: Input element type
    - R: Result type
    """
    func: Literal["COUNT", "SUM", "AVG", "MIN", "MAX"]
    column: Ref[Node[T]]

# Examples
count_users = Aggregate[Any, int](
    func="COUNT",
    column=Ref(id="id_col")  # COUNT returns int
)

total_amount = Aggregate[float, float](
    func="SUM",
    column=Ref(id="amount_col")  # SUM of float returns float
)

avg_age = Aggregate[int, float](
    func="AVG",
    column=Ref(id="age_col")  # AVG of int returns float
)
```

### Extend Generator

```python
# In SQLGenerator.eval():
case Aggregate(func=f, column=c):
    col_sql = self.eval(self.resolve(c))
    return f"{f}({col_sql})"
```

### Type Safety Benefits

```python
# ✓ Type-safe: SUM on numeric column
total = Aggregate[float, float](
    func="SUM",
    column=amount_column  # Column[float]
)

# ✗ Type error: SUM on string column
broken = Aggregate[str, ???](  # What result type?
    func="SUM",
    column=name_column  # Column[str] - can't sum strings!
)
```

---

## Step 5: Complete Example (8 min)

### Complex Query with Multiple Features

```python
# Build:
# SELECT name, email, age
# FROM users
# WHERE age >= 18 AND active = TRUE
# LIMIT 100

ast = AST(
    root="query",
    nodes={
        # age >= 18
        "age_col": Column[int](table=None, name="age"),
        "age_val": Literal[int](value=18),
        "age_check": Comparison[int](
            op=">=",
            left=Ref(id="age_col"),
            right=Ref(id="age_val")
        ),

        # active = TRUE
        "active_col": Column[bool](table=None, name="active"),
        "true_val": Literal[bool](value=True),
        "active_check": Comparison[bool](
            op="=",
            left=Ref(id="active_col"),
            right=Ref(id="true_val")
        ),

        # age >= 18 AND active = TRUE
        "combined": LogicalOp(
            op="AND",
            left=Ref(id="age_check"),
            right=Ref(id="active_check")
        ),

        # SELECT query
        "query": Select(
            columns=["name", "email", "age"],
            table="users",
            where=Ref(id="combined"),
            limit=100
        )
    }
)

print("=== Validation ===")
validator = QueryValidator(ast, db_schema)
validator.run()
print("✓ Query is valid")

print("\n=== SQL Generation ===")
generator = SQLGenerator(ast, None)
sql = generator.run()
print(sql)
# Output: SELECT name, email, age FROM users WHERE (age >= 18 AND active = TRUE) LIMIT 100

print("\n=== Serialization ===")
json_str = ast.to_json()
print(f"Serialized to {len(json_str)} bytes")

print("\n=== Round-Trip ===")
restored_ast = AST.from_json(json_str)
restored_sql = SQLGenerator(restored_ast, None).run()
assert restored_sql == sql
print("✓ Round-trip successful")
```

---

## Extensions

### 1. JOINs

```python
class Join(Node[list[dict]], tag="sql_join"):
    join_type: Literal["INNER", "LEFT", "RIGHT", "OUTER"]
    left_table: str
    right_table: str
    on_condition: Ref[Node[bool]]
```

### 2. Subqueries

```python
class Subquery(Node[list[dict]], tag="sql_subquery"):
    query: Ref[Select]
    alias: str
```

### 3. GROUP BY

```python
class GroupBy(Node[list[dict]], tag="sql_groupby"):
    query: Ref[Select]
    group_columns: list[str]
    having: Ref[Node[bool]] | None = None
```

### 4. Query Optimization

Create an optimizer interpreter:

```python
class QueryOptimizer(Interpreter[None, AST]):
    """Optimize queries by transforming AST."""

    def eval(self, node: Node[Any]) -> Node[Any]:
        match node:
            # Push down filters
            # Combine adjacent filters
            # Eliminate redundant conditions
            ...
```

---

## Key Takeaways

✓ **Type parameters** enable type-safe comparisons and aggregations
✓ **Multiple interpreters** handle validation, generation, and optimization
✓ **AST structure** prevents SQL injection automatically
✓ **Schema validation** catches errors before database execution
✓ **Serialization** enables query storage and transmission

---

## Real-World Applications

This pattern applies to:
- **ORM query builders**: Type-safe database queries
- **GraphQL query construction**: Build and validate GraphQL queries
- **API request builders**: Type-safe API client generation
- **Config generators**: Generate configuration from validated ASTs

---

## Next Steps

- Add support for JOINs, subqueries, GROUP BY
- Implement query optimization passes
- Add database-specific dialects (PostgreSQL, MySQL, SQLite)
- Connect to real database and execute queries
- Build a query planner and optimizer
