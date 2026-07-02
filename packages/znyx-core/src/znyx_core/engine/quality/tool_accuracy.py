"""Tool call accuracy scorer - validates tool arguments against declared schemas.

Checks required params, types, enum values, and value ranges.
Only meaningful in tool evaluation context.
"""
from typing import Dict, Any, Optional

from znyx_core.core.models import QualityScore


def _validate_field(value: Any, schema: Dict[str, Any]) -> bool:
    """Check if a value satisfies a JSON Schema field definition."""
    expected_type = schema.get("type")

    if expected_type == "string":
        if not isinstance(value, str):
            return False
        if "enum" in schema and value not in schema["enum"]:
            return False
        if "minLength" in schema and len(value) < schema["minLength"]:
            return False
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            return False
        if "pattern" in schema:
            import re
            if not re.match(schema["pattern"], value):
                return False
    elif expected_type == "number" or expected_type == "integer":
        if not isinstance(value, (int, float)):
            return False
        if expected_type == "integer" and not isinstance(value, int):
            return False
        if "minimum" in schema and value < schema["minimum"]:
            return False
        if "maximum" in schema and value > schema["maximum"]:
            return False
        if "enum" in schema and value not in schema["enum"]:
            return False
    elif expected_type == "boolean":
        if not isinstance(value, bool):
            return False
    elif expected_type == "array":
        if not isinstance(value, list):
            return False
        if "minItems" in schema and len(value) < schema["minItems"]:
            return False
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            return False
    elif expected_type == "object":
        if not isinstance(value, dict):
            return False

    return True


def score_tool_accuracy(
    tool_context: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> QualityScore:
    """Score tool call accuracy against declared schema.

    Expects tool_context with keys: tool_name, tool_args, tool_schema.
    tool_schema follows JSON Schema format with properties and required.
    """
    if not tool_context:
        return QualityScore(
            metric="tool_call_accuracy",
            score=1.0,
            details="No tool context provided; score defaults to 1.0.",
        )

    tool_args = tool_context.get("tool_args", {})
    tool_schema = tool_context.get("tool_schema", {})

    if not tool_schema:
        # No schema to validate against
        return QualityScore(
            metric="tool_call_accuracy",
            score=1.0,
            details="No tool schema provided; score defaults to 1.0.",
        )

    properties = tool_schema.get("properties", {})
    required = set(tool_schema.get("required", []))

    if not properties:
        return QualityScore(
            metric="tool_call_accuracy",
            score=1.0,
            details="Empty schema properties; score defaults to 1.0.",
        )

    total_checks = len(properties)
    valid_count = 0

    # Check required fields presence
    missing_required = required - set(tool_args.keys())

    for field_name, field_schema in properties.items():
        if field_name in tool_args:
            if _validate_field(tool_args[field_name], field_schema):
                valid_count += 1
            # else: field present but invalid type/value
        elif field_name not in required:
            valid_count += 1  # optional and absent is fine
        # else: required and missing, already counted

    score = valid_count / total_checks if total_checks > 0 else 1.0
    score = min(max(score, 0.0), 1.0)

    details_parts = []
    if missing_required:
        details_parts.append(f"missing_required={list(missing_required)}")
    details_parts.append(f"{valid_count}/{total_checks} fields valid")

    return QualityScore(
        metric="tool_call_accuracy",
        score=round(score, 3),
        details=", ".join(details_parts),
        sub_scores={
            "valid_fields": valid_count,
            "total_fields": total_checks,
            "missing_required": len(missing_required),
        },
    )
