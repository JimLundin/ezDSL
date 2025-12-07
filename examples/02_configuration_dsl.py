"""
Configuration DSL Example
==========================

This example demonstrates how to build a configuration DSL with validation.
It covers:

1. Modeling configuration structures as AST nodes
2. Implementing custom validation logic
3. Building interpreters that validate and process configurations
4. Handling errors and providing helpful error messages
5. Using custom validation contexts

The configuration DSL supports:
- String, integer, and boolean settings
- Nested configuration groups
- Environment variable references
- Validation rules (required fields, ranges, patterns)
"""

from dataclasses import dataclass
from typing import Literal, Any
import re

from nanodsl import Node, AST, Ref, Interpreter


# ============================================================================
# Step 1: Define Configuration Nodes
# ============================================================================


class StringValue(Node[str], tag="cfg_string"):
    """A string configuration value."""

    value: str


class IntValue(Node[int], tag="cfg_int"):
    """An integer configuration value."""

    value: int


class BoolValue(Node[bool], tag="cfg_bool"):
    """A boolean configuration value."""

    value: bool


class EnvVar(Node[str], tag="cfg_env"):
    """
    Reference to an environment variable.

    Example: EnvVar(name="HOME", default="/tmp")
    """

    name: str
    default: str | None = None


class ConfigGroup(Node[dict[str, Any]], tag="cfg_group"):
    """
    A group of configuration settings.

    Fields are references to other configuration nodes.
    """

    name: str
    settings: dict[str, Ref[Node[Any]]]


# ============================================================================
# Step 2: Define Validation Rules
# ============================================================================
# Validation rules are not nodes themselves, but metadata for validation.


@dataclass(frozen=True)
class ValidationRule:
    """Base class for validation rules."""

    field_path: str  # Dot-separated path to field (e.g., "database.port")


@dataclass(frozen=True)
class RequiredRule(ValidationRule):
    """Field must be present and non-empty."""

    pass


@dataclass(frozen=True)
class RangeRule(ValidationRule):
    """Integer field must be within a range."""

    min_value: int | None = None
    max_value: int | None = None


@dataclass(frozen=True)
class PatternRule(ValidationRule):
    """String field must match a regex pattern."""

    pattern: str


@dataclass(frozen=True)
class EnumRule(ValidationRule):
    """Field value must be one of the allowed values."""

    allowed_values: tuple[str, ...]


# ============================================================================
# Step 3: Validation Context
# ============================================================================


@dataclass
class ValidationContext:
    """
    Context for configuration validation and evaluation.

    Attributes:
        env_vars: Environment variables available for substitution
        validation_rules: Rules to check during evaluation
        errors: Accumulated validation errors
    """

    env_vars: dict[str, str]
    validation_rules: list[ValidationRule]
    errors: list[str]

    def add_error(self, error: str) -> None:
        """Add a validation error."""
        self.errors.append(error)

    def has_errors(self) -> bool:
        """Check if any validation errors were found."""
        return len(self.errors) > 0


# ============================================================================
# Step 4: Configuration Validator/Interpreter
# ============================================================================


class ConfigInterpreter(Interpreter[ValidationContext, dict[str, Any]]):
    """
    Interprets and validates configuration DSL.

    This interpreter:
    1. Evaluates configuration nodes to produce values
    2. Resolves environment variable references
    3. Validates against rules in the context
    4. Accumulates errors for reporting
    """

    def eval(self, node: Node[Any]) -> Any:
        """Evaluate a configuration node."""
        match node:
            case StringValue(value=v):
                return v

            case IntValue(value=v):
                return v

            case BoolValue(value=v):
                return v

            case EnvVar(name=n, default=d):
                # Look up environment variable
                if n in self.ctx.env_vars:
                    return self.ctx.env_vars[n]
                elif d is not None:
                    return d
                else:
                    self.ctx.add_error(
                        f"Environment variable '{n}' not found and no default provided"
                    )
                    return None

            case ConfigGroup(name=group_name, settings=settings_dict):
                # Recursively evaluate all settings in the group
                result = {}
                for key, ref in settings_dict.items():
                    setting_node = self.resolve(ref)
                    result[key] = self.eval(setting_node)
                return result

            case _:
                raise NotImplementedError(f"Unknown node type: {type(node)}")

    def validate_rules(self, config: dict[str, Any]) -> None:
        """
        Validate the evaluated configuration against rules.

        This is called after evaluation to check business rules.
        """
        for rule in self.ctx.validation_rules:
            self._validate_single_rule(rule, config)

    def _validate_single_rule(
        self, rule: ValidationRule, config: dict[str, Any]
    ) -> None:
        """Validate a single rule against the configuration."""
        # Navigate to the field using the path
        path_parts = rule.field_path.split(".")
        value = config

        try:
            for part in path_parts:
                value = value[part]
        except (KeyError, TypeError):
            if isinstance(rule, RequiredRule):
                self.ctx.add_error(f"Required field '{rule.field_path}' is missing")
            return

        # Apply type-specific validation
        match rule:
            case RequiredRule():
                if value is None or value == "":
                    self.ctx.add_error(
                        f"Required field '{rule.field_path}' is empty"
                    )

            case RangeRule(min_value=min_val, max_value=max_val):
                if not isinstance(value, int):
                    self.ctx.add_error(
                        f"Field '{rule.field_path}' must be an integer"
                    )
                elif min_val is not None and value < min_val:
                    self.ctx.add_error(
                        f"Field '{rule.field_path}' value {value} is below minimum {min_val}"
                    )
                elif max_val is not None and value > max_val:
                    self.ctx.add_error(
                        f"Field '{rule.field_path}' value {value} exceeds maximum {max_val}"
                    )

            case PatternRule(pattern=pattern):
                if not isinstance(value, str):
                    self.ctx.add_error(
                        f"Field '{rule.field_path}' must be a string"
                    )
                elif not re.match(pattern, value):
                    self.ctx.add_error(
                        f"Field '{rule.field_path}' value '{value}' does not match pattern '{pattern}'"
                    )

            case EnumRule(allowed_values=allowed):
                if value not in allowed:
                    self.ctx.add_error(
                        f"Field '{rule.field_path}' value '{value}' is not one of {allowed}"
                    )


# ============================================================================
# Step 5: Example Usage
# ============================================================================


def example_valid_configuration():
    """Example of a valid configuration."""

    # Build configuration AST
    ast = AST(
        root="config",
        nodes={
            # Database configuration
            "db_host": StringValue(value="localhost"),
            "db_port": IntValue(value=5432),
            "db_name": StringValue(value="myapp"),
            "db_user": EnvVar(name="DB_USER", default="postgres"),
            "db_password": EnvVar(name="DB_PASSWORD", default=None),
            "database": ConfigGroup(
                name="database",
                settings={
                    "host": Ref(id="db_host"),
                    "port": Ref(id="db_port"),
                    "name": Ref(id="db_name"),
                    "user": Ref(id="db_user"),
                    "password": Ref(id="db_password"),
                },
            ),
            # Server configuration
            "server_port": IntValue(value=8080),
            "server_host": StringValue(value="0.0.0.0"),
            "debug": BoolValue(value=False),
            "server": ConfigGroup(
                name="server",
                settings={
                    "port": Ref(id="server_port"),
                    "host": Ref(id="server_host"),
                    "debug": Ref(id="debug"),
                },
            ),
            # Root configuration
            "config": ConfigGroup(
                name="root",
                settings={
                    "database": Ref(id="database"),
                    "server": Ref(id="server"),
                },
            ),
        },
    )

    # Define validation rules
    validation_rules = [
        RequiredRule(field_path="database.host"),
        RequiredRule(field_path="database.port"),
        RequiredRule(field_path="database.password"),
        RangeRule(field_path="database.port", min_value=1024, max_value=65535),
        RangeRule(field_path="server.port", min_value=1024, max_value=65535),
        PatternRule(
            field_path="database.host", pattern=r"^[a-zA-Z0-9.-]+$"
        ),  # Hostname pattern
    ]

    # Create validation context
    ctx = ValidationContext(
        env_vars={
            "DB_USER": "admin",
            "DB_PASSWORD": "secret123",
        },
        validation_rules=validation_rules,
        errors=[],
    )

    # Evaluate and validate
    interpreter = ConfigInterpreter(ast, ctx)
    config = interpreter.run()

    # Run additional validation
    interpreter.validate_rules(config)

    print("=" * 80)
    print("Valid Configuration Example")
    print("=" * 80)
    print()
    print("Evaluated configuration:")
    print(config)
    print()

    if ctx.has_errors():
        print("Validation errors:")
        for error in ctx.errors:
            print(f"  ✗ {error}")
    else:
        print("✓ Configuration is valid!")

    print()
    return config, ctx


def example_invalid_configuration():
    """Example of an invalid configuration with validation errors."""

    # Build configuration with problems
    ast = AST(
        root="config",
        nodes={
            "db_host": StringValue(value="invalid host name!"),  # Invalid characters
            "db_port": IntValue(value=99999),  # Out of range
            "db_password": EnvVar(
                name="DB_PASSWORD", default=None
            ),  # Missing env var, no default
            "database": ConfigGroup(
                name="database",
                settings={
                    "host": Ref(id="db_host"),
                    "port": Ref(id="db_port"),
                    "password": Ref(id="db_password"),
                },
            ),
            "config": ConfigGroup(
                name="root",
                settings={
                    "database": Ref(id="database"),
                },
            ),
        },
    )

    # Define validation rules (same as before)
    validation_rules = [
        RequiredRule(field_path="database.password"),
        RangeRule(field_path="database.port", min_value=1024, max_value=65535),
        PatternRule(field_path="database.host", pattern=r"^[a-zA-Z0-9.-]+$"),
    ]

    # Create validation context (no env vars provided)
    ctx = ValidationContext(
        env_vars={},  # Empty - will cause DB_PASSWORD to fail
        validation_rules=validation_rules,
        errors=[],
    )

    # Evaluate and validate
    interpreter = ConfigInterpreter(ast, ctx)
    config = interpreter.run()

    # Run additional validation
    interpreter.validate_rules(config)

    print("=" * 80)
    print("Invalid Configuration Example")
    print("=" * 80)
    print()
    print("Evaluated configuration:")
    print(config)
    print()

    if ctx.has_errors():
        print(f"Found {len(ctx.errors)} validation error(s):")
        for i, error in enumerate(ctx.errors, 1):
            print(f"  {i}. ✗ {error}")
    else:
        print("✓ Configuration is valid!")

    print()
    return config, ctx


def example_custom_validation():
    """Example showing how to implement custom validation logic."""

    # Build a simple config
    ast = AST(
        root="config",
        nodes={
            "min_val": IntValue(value=10),
            "max_val": IntValue(value=5),  # Problem: max < min!
            "config": ConfigGroup(
                name="range",
                settings={
                    "min": Ref(id="min_val"),
                    "max": Ref(id="max_val"),
                },
            ),
        },
    )

    ctx = ValidationContext(env_vars={}, validation_rules=[], errors=[])

    # Evaluate
    interpreter = ConfigInterpreter(ast, ctx)
    config = interpreter.run()

    # Custom validation: ensure max >= min
    if config["min"] > config["max"]:
        ctx.add_error(f"Invalid range: min ({config['min']}) > max ({config['max']})")

    print("=" * 80)
    print("Custom Validation Example")
    print("=" * 80)
    print()
    print("Configuration:")
    print(config)
    print()

    if ctx.has_errors():
        print("Validation errors:")
        for error in ctx.errors:
            print(f"  ✗ {error}")
    else:
        print("✓ Configuration is valid!")

    print()
    return config, ctx


# ============================================================================
# Main: Run All Examples
# ============================================================================


def main():
    """Run all configuration DSL examples."""

    example_valid_configuration()
    example_invalid_configuration()
    example_custom_validation()

    print("All examples completed! ✓")


if __name__ == "__main__":
    main()
