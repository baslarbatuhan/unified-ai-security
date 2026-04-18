"""
output_agency_defense/parameter_validation.py
===============================================
Tool parameter validation guard.

Purpose:
    Validate tool call parameters before execution:
    1. Required params present
    2. Correct types (string, int, etc.)
    3. Format validation (no SQL injection, path traversal, overflow)
    4. Suspicious character detection

Usage:
    validator = ParameterValidator()
    validator.register_tool_schema("get_order", {"resource_id": {"type": "str", "required": True, "max_length": 50, "pattern": "^[A-Z]+-[0-9]+$"}})
    result = validator.validate("get_order", {"resource_id": "ORD-001"})
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

Decision = Literal["allow", "block"]

# Suspicious patterns in parameter values
SUSPICIOUS_PATTERNS = [
    (re.compile(r"['\";].*(--)"), "sql_injection"),
    (re.compile(r"(\.\./|\.\.\\)"), "path_traversal"),
    (re.compile(r"<script", re.I), "xss"),
    (re.compile(r"(\$\{|`.*`|(?<!\w)[;&|<>]|\$\(|&&|\|\|)"), "command_injection"),
    (re.compile(r"__import__\s*\("), "python_injection"),
    (re.compile(r"(\x00|\x0a|\x0d)"), "null_byte"),
]


@dataclass
class ParamSchema:
    name: str
    type: str = "str"       # str, int, float, bool
    required: bool = True
    max_length: int = 200
    pattern: Optional[str] = None  # regex for format validation
    allowed_values: Optional[List] = None   # allowlist: only these values accepted
    denied_values: Optional[List] = None    # denylist: these values always rejected
    min_value: Optional[float] = None       # numeric min (inclusive)
    max_value: Optional[float] = None       # numeric max (inclusive)


@dataclass
class ValidationResult:
    decision: Decision
    tool: str
    is_valid: bool = True
    violations: List[str] = field(default_factory=list)
    risk_contribution: float = 0.0

    @property
    def is_blocked(self) -> bool:
        return self.decision == "block"


class ParameterValidator:
    """
    Validates tool parameters for type, format, and suspicious content.

    Checks:
    - Required parameters present
    - Type correctness (str, int, float, bool)
    - Max length (buffer overflow prevention)
    - Format pattern (regex)
    - Suspicious characters (SQL injection, path traversal, XSS, etc.)
    """

    def __init__(self):
        self._schemas: Dict[str, List[ParamSchema]] = {}

    def register_tool_schema(self, tool_name: str, params: Dict[str, Dict]):
        """Register parameter schema for a tool.

        Each param spec supports:
            type, required, max_length, pattern,
            allowed_values (allowlist), denied_values (denylist)
        """
        schemas = []
        for name, spec in params.items():
            schemas.append(ParamSchema(
                name=name,
                type=spec.get("type", "str"),
                required=spec.get("required", True),
                max_length=spec.get("max_length", 200),
                pattern=spec.get("pattern"),
                allowed_values=spec.get("allowed_values"),
                denied_values=spec.get("denied_values"),
                min_value=spec.get("min_value"),
                max_value=spec.get("max_value"),
            ))
        self._schemas[tool_name] = schemas

    def validate(self, tool_name: str, args: Dict[str, Any]) -> ValidationResult:
        """
        Validate parameters against registered schema.

        Checks (in order):
        1. Unexpected parameters (not in schema) → rejected
        2. Required parameters present
        3. Type correctness
        4. String: max_length, format pattern, suspicious content
        5. Denylist (always rejected values)
        6. Allowlist (only accepted values)

        Returns ValidationResult with violations list.
        """
        violations = []

        schemas = self._schemas.get(tool_name)
        if schemas is None:
            return ValidationResult(decision="allow", tool=tool_name, is_valid=True,
                                    violations=["No schema registered, skipping validation"])

        # --- Unexpected parameter rejection ---
        known_names = {s.name for s in schemas}
        unexpected = set(args.keys()) - known_names
        for param_name in sorted(unexpected):
            violations.append(f"Unexpected parameter: '{param_name}' is not in tool schema")

        for schema in schemas:
            value = args.get(schema.name)

            # Required check
            if schema.required and (value is None or value == ""):
                violations.append(f"Missing required parameter: '{schema.name}'")
                continue

            if value is None:
                continue

            # Type check
            expected_types = {"str": str, "int": int, "float": (int, float), "bool": bool}
            expected = expected_types.get(schema.type)
            if expected and not isinstance(value, expected):
                violations.append(f"Wrong type for '{schema.name}': expected {schema.type}, got {type(value).__name__}")
                continue

            # Numeric range checks
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if schema.min_value is not None and value < schema.min_value:
                    violations.append(f"Parameter '{schema.name}' below minimum: {value} < {schema.min_value}")
                if schema.max_value is not None and value > schema.max_value:
                    violations.append(f"Parameter '{schema.name}' above maximum: {value} > {schema.max_value}")

            # String-specific checks
            if isinstance(value, str):
                # Max length
                if len(value) > schema.max_length:
                    violations.append(f"Parameter '{schema.name}' exceeds max length: {len(value)} > {schema.max_length}")

                # Format pattern
                if schema.pattern and not re.match(schema.pattern, value):
                    violations.append(f"Parameter '{schema.name}' format invalid: does not match {schema.pattern}")

                # Suspicious content
                for pattern, attack_type in SUSPICIOUS_PATTERNS:
                    if pattern.search(value):
                        violations.append(f"Suspicious content in '{schema.name}': {attack_type} detected")

            # Denylist check (before allowlist — explicit deny always wins)
            if schema.denied_values and value in schema.denied_values:
                violations.append(f"Parameter '{schema.name}' contains denied value: '{value}'")

            # Allowlist check
            if schema.allowed_values and value not in schema.allowed_values:
                violations.append(f"Parameter '{schema.name}' value not allowed: '{value}'")

        is_valid = len(violations) == 0
        risk = 0.0 if is_valid else min(0.5 + len(violations) * 0.15, 1.0)

        return ValidationResult(
            decision="allow" if is_valid else "block",
            tool=tool_name,
            is_valid=is_valid,
            violations=violations,
            risk_contribution=round(risk, 4),
        )


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    validator = ParameterValidator()

    # Register schemas
    validator.register_tool_schema("get_order", {
        "resource_id": {"type": "str", "required": True, "max_length": 50, "pattern": r"^[A-Z]+-\d+$"},
    })
    validator.register_tool_schema("update_ticket", {
        "resource_id": {"type": "str", "required": True, "max_length": 50, "pattern": r"^[A-Z]+-\d+$"},
        "status": {
            "type": "str", "required": True,
            "allowed_values": ["open", "in_progress", "closed"],
            "denied_values": ["deleted", "purged"],
        },
    })

    tests = [
        ("Valid order", "get_order", {"resource_id": "ORD-001"}),
        ("SQL injection", "get_order", {"resource_id": "ORD-001'; DROP TABLE orders;--"}),
        ("Path traversal", "get_order", {"resource_id": "../../etc/passwd"}),
        ("Missing param", "get_order", {}),
        ("Wrong type", "get_order", {"resource_id": 12345}),
        ("Too long", "get_order", {"resource_id": "ORD-" + "A" * 200}),
        ("Valid ticket update", "update_ticket", {"resource_id": "TKT-101", "status": "closed"}),
        ("Invalid status", "update_ticket", {"resource_id": "TKT-101", "status": "deleted"}),
        ("Denied status", "update_ticket", {"resource_id": "TKT-101", "status": "purged"}),
        ("Unexpected param", "get_order", {"resource_id": "ORD-001", "debug": "true", "admin": "1"}),
        ("Null resource_id", "get_order", {"resource_id": None}),
    ]

    print(f"{'='*60}\n  PARAMETER VALIDATION DEMO\n{'='*60}")
    for desc, tool, args in tests:
        r = validator.validate(tool, args)
        status = "VALID" if r.is_valid else "BLOCKED"
        print(f"\n  [{status:7s}] {desc}")
        print(f"    Tool: {tool} | Args: {args}")
        if r.violations:
            for v in r.violations:
                print(f"    Violation: {v}")
    print(f"\n{'='*60}")
