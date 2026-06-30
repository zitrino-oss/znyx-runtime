"""
Output Structure Enforcement Detector

Validates output format (JSON/XML) with full JSON Schema Draft 7 support
and field-level error reporting for structured output contracts.
"""
import json
import re
import xml.etree.ElementTree as ET
# Untrusted model output is parsed here: defusedxml hardens against XXE and
# entity-expansion ("billion laughs") DoS that the stdlib parser is exposed to.
from defusedxml.ElementTree import fromstring as safe_xml_fromstring
from defusedxml.common import DefusedXmlException
from typing import List, Dict, Any, Optional
from app.shared.core.models import DetectorResult, RuleHit, Severity, Decision, FieldError


class StructureDetector:
    """Detects and validates output structure (JSON/XML) with schema contracts."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get('enabled', False)
        self.action = config.get('action', 'BLOCK')
        self.expected_format = config.get('expected_format')
        # Accept both 'json_schema' and 'schema' as aliases
        self.json_schema = config.get('json_schema') or config.get('schema')
        self.xml_root = config.get('xml_root')
        # Named schema support: resolve from schema_name if provided
        self.schema_name = config.get('schema_name')
        # Auto-infer format from sibling keys when not explicitly set
        if not self.expected_format:
            if self.json_schema or self.schema_name:
                self.expected_format = 'json'
            elif self.xml_root:
                self.expected_format = 'xml'
            elif self.enabled:
                self.expected_format = 'json'

    def _validate_json(self, text: str) -> tuple:
        """Validate JSON format and schema. Returns (rule_hits, field_errors)."""
        rule_hits = []
        field_errors = []

        # Extract JSON from markdown code blocks if present
        cleaned = text.strip()
        md_match = re.search(r'```(?:json)?\s*\n(.*?)\n```', cleaned, re.DOTALL)
        if md_match:
            cleaned = md_match.group(1).strip()

        try:
            data = json.loads(cleaned)
        except (json.JSONDecodeError, ValueError) as e:
            rule_hits.append(RuleHit(
                rule_id="structure.json_invalid",
                severity=Severity.MEDIUM,
                message=f"Invalid JSON format: {str(e)}"
            ))
            field_errors.append(FieldError(
                path="/",
                message=f"JSON parse error: {str(e)}",
            ))
            return rule_hits, field_errors

        # Full JSON Schema validation
        if self.json_schema:
            schema_errors = _validate_schema(data, self.json_schema)
            if schema_errors:
                rule_hits.append(RuleHit(
                    rule_id="structure.json_schema_invalid",
                    severity=Severity.MEDIUM,
                    message=f"JSON schema validation failed: {len(schema_errors)} error(s)"
                ))
                field_errors.extend(schema_errors)

        return rule_hits, field_errors

    def _validate_xml(self, text: str) -> tuple:
        """Validate XML format and optional root tag. Returns (rule_hits, field_errors)."""
        rule_hits = []
        field_errors = []

        try:
            root = safe_xml_fromstring(text)
        except (ET.ParseError, DefusedXmlException) as e:
            rule_hits.append(RuleHit(
                rule_id="structure.xml_invalid",
                severity=Severity.MEDIUM,
                message=f"Invalid XML format: {str(e)}"
            ))
            field_errors.append(FieldError(path="/", message=f"XML parse error: {str(e)}"))
            return rule_hits, field_errors

        if self.xml_root and root.tag != self.xml_root:
            rule_hits.append(RuleHit(
                rule_id="structure.xml_root_mismatch",
                severity=Severity.MEDIUM,
                message=f"Expected root tag '{self.xml_root}', got '{root.tag}'"
            ))
            field_errors.append(FieldError(
                path="/",
                message=f"Root tag mismatch",
                expected=self.xml_root,
                actual=root.tag,
            ))

        return rule_hits, field_errors

    def detect(self, text: str) -> DetectorResult:
        """Validate output structure with field-level error reporting."""
        if not self.enabled or not self.expected_format:
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)

        rule_hits = []
        field_errors = []

        if self.expected_format == 'json':
            rule_hits, field_errors = self._validate_json(text)
        elif self.expected_format == 'xml':
            rule_hits, field_errors = self._validate_xml(text)

        if not rule_hits:
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)

        decision = Decision.BLOCK if self.action == 'BLOCK' else Decision.WARN

        return DetectorResult(
            decision=decision,
            risk_score=30,
            rule_hits=rule_hits,
            field_errors=field_errors,
            sanitized_text=None,
            developer_message=f"structure: {len(field_errors)} field error(s). "
                              f"Output does not conform to required {self.expected_format.upper()} schema."
        )


# -- JSON Schema Draft 7 validation ------------------------------------------

def _validate_schema(data: Any, schema: Dict[str, Any], path: str = "") -> List[FieldError]:
    """Validate data against a JSON Schema Draft 7 definition.

    Returns a list of FieldError for each violation found.
    """
    errors: List[FieldError] = []
    schema_type = schema.get("type")

    # Type check
    if schema_type:
        if not _check_type(data, schema_type):
            errors.append(FieldError(
                path=path or "/",
                message=f"Expected type '{schema_type}', got '{type(data).__name__}'",
                expected=schema_type,
                actual=type(data).__name__,
            ))
            return errors  # Can't drill deeper with wrong type

    # Enum check
    if "enum" in schema:
        if data not in schema["enum"]:
            errors.append(FieldError(
                path=path or "/",
                message=f"Value not in allowed enum: {schema['enum']}",
                expected=str(schema["enum"]),
                actual=str(data),
            ))

    # Const check
    if "const" in schema:
        if data != schema["const"]:
            errors.append(FieldError(
                path=path or "/",
                message=f"Value must be {schema['const']}",
                expected=str(schema["const"]),
                actual=str(data),
            ))

    # Object validation
    if isinstance(data, dict):
        errors.extend(_validate_object(data, schema, path))

    # Array validation
    if isinstance(data, list):
        errors.extend(_validate_array(data, schema, path))

    # String validation
    if isinstance(data, str):
        errors.extend(_validate_string(data, schema, path))

    # Number validation
    if isinstance(data, (int, float)) and not isinstance(data, bool):
        errors.extend(_validate_number(data, schema, path))

    # Combinators
    if "allOf" in schema:
        for sub in schema["allOf"]:
            errors.extend(_validate_schema(data, sub, path))
    if "anyOf" in schema:
        if not any(len(_validate_schema(data, sub, path)) == 0 for sub in schema["anyOf"]):
            errors.append(FieldError(path=path or "/", message="Value does not match any of the 'anyOf' schemas"))
    if "oneOf" in schema:
        matches = sum(1 for sub in schema["oneOf"] if len(_validate_schema(data, sub, path)) == 0)
        if matches != 1:
            errors.append(FieldError(path=path or "/", message=f"Value must match exactly one 'oneOf' schema, matched {matches}"))
    if "not" in schema:
        if len(_validate_schema(data, schema["not"], path)) == 0:
            errors.append(FieldError(path=path or "/", message="Value must NOT match the 'not' schema"))

    # if/then/else
    if "if" in schema:
        if_errors = _validate_schema(data, schema["if"], path)
        if len(if_errors) == 0 and "then" in schema:
            errors.extend(_validate_schema(data, schema["then"], path))
        elif len(if_errors) > 0 and "else" in schema:
            errors.extend(_validate_schema(data, schema["else"], path))

    return errors


def _check_type(data: Any, schema_type: str) -> bool:
    type_map = {
        "string": str,
        "number": (int, float),
        "integer": int,
        "boolean": bool,
        "array": list,
        "object": dict,
        "null": type(None),
    }
    expected = type_map.get(schema_type)
    if expected is None:
        return True
    if schema_type == "number" and isinstance(data, bool):
        return False
    if schema_type == "integer" and isinstance(data, bool):
        return False
    return isinstance(data, expected)


def _validate_object(data: dict, schema: Dict[str, Any], path: str) -> List[FieldError]:
    errors = []
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    additional = schema.get("additionalProperties")
    pattern_props = schema.get("patternProperties", {})
    min_props = schema.get("minProperties")
    max_props = schema.get("maxProperties")

    # Required check
    for field in required:
        if field not in data:
            errors.append(FieldError(
                path=f"{path}/{field}",
                message=f"Required field '{field}' is missing",
            ))

    # Validate known properties
    for prop, prop_schema in properties.items():
        if prop in data:
            errors.extend(_validate_schema(data[prop], prop_schema, f"{path}/{prop}"))

    # Pattern properties
    for pattern, prop_schema in pattern_props.items():
        for key in data:
            if re.match(pattern, key):
                errors.extend(_validate_schema(data[key], prop_schema, f"{path}/{key}"))

    # Additional properties
    if additional is False:
        known = set(properties.keys())
        for key in data:
            if key not in known:
                matched_pattern = any(re.match(p, key) for p in pattern_props)
                if not matched_pattern:
                    errors.append(FieldError(
                        path=f"{path}/{key}",
                        message=f"Additional property '{key}' is not allowed",
                    ))
    elif isinstance(additional, dict):
        known = set(properties.keys())
        for key in data:
            if key not in known:
                errors.extend(_validate_schema(data[key], additional, f"{path}/{key}"))

    # Property count
    if min_props is not None and len(data) < min_props:
        errors.append(FieldError(path=path or "/", message=f"Object has {len(data)} properties, minimum is {min_props}"))
    if max_props is not None and len(data) > max_props:
        errors.append(FieldError(path=path or "/", message=f"Object has {len(data)} properties, maximum is {max_props}"))

    # Dependencies
    for dep_key, dep_value in schema.get("dependencies", {}).items():
        if dep_key in data:
            if isinstance(dep_value, list):
                for req in dep_value:
                    if req not in data:
                        errors.append(FieldError(
                            path=f"{path}/{req}",
                            message=f"Property '{req}' is required when '{dep_key}' is present",
                        ))
            elif isinstance(dep_value, dict):
                errors.extend(_validate_schema(data, dep_value, path))

    return errors


def _validate_array(data: list, schema: Dict[str, Any], path: str) -> List[FieldError]:
    errors = []
    items = schema.get("items")
    min_items = schema.get("minItems")
    max_items = schema.get("maxItems")
    unique = schema.get("uniqueItems", False)
    contains = schema.get("contains")

    if min_items is not None and len(data) < min_items:
        errors.append(FieldError(path=path or "/", message=f"Array has {len(data)} items, minimum is {min_items}"))
    if max_items is not None and len(data) > max_items:
        errors.append(FieldError(path=path or "/", message=f"Array has {len(data)} items, maximum is {max_items}"))

    if unique:
        seen = []
        for i, item in enumerate(data):
            key = json.dumps(item, sort_keys=True, default=str)
            if key in seen:
                errors.append(FieldError(path=f"{path}[{i}]", message="Duplicate item in array with uniqueItems"))
                break
            seen.append(key)

    if items:
        if isinstance(items, dict):
            for i, item in enumerate(data):
                errors.extend(_validate_schema(item, items, f"{path}[{i}]"))
        elif isinstance(items, list):
            # Tuple validation
            for i, item_schema in enumerate(items):
                if i < len(data):
                    errors.extend(_validate_schema(data[i], item_schema, f"{path}[{i}]"))

    if contains:
        if not any(len(_validate_schema(item, contains, "")) == 0 for item in data):
            errors.append(FieldError(path=path or "/", message="No item matches 'contains' schema"))

    return errors


def _validate_string(data: str, schema: Dict[str, Any], path: str) -> List[FieldError]:
    errors = []
    if "minLength" in schema and len(data) < schema["minLength"]:
        errors.append(FieldError(path=path or "/", message=f"String length {len(data)} below minimum {schema['minLength']}"))
    if "maxLength" in schema and len(data) > schema["maxLength"]:
        errors.append(FieldError(path=path or "/", message=f"String length {len(data)} exceeds maximum {schema['maxLength']}"))
    if "pattern" in schema and not re.search(schema["pattern"], data):
        errors.append(FieldError(path=path or "/", message=f"String does not match pattern '{schema['pattern']}'"))

    # Format validation (common formats)
    fmt = schema.get("format")
    if fmt == "email" and not re.match(r"^[^@]+@[^@]+\.[^@]+$", data):
        errors.append(FieldError(path=path or "/", message="Invalid email format"))
    elif fmt == "uri" and not re.match(r"^https?://", data):
        errors.append(FieldError(path=path or "/", message="Invalid URI format"))
    elif fmt == "date" and not re.match(r"^\d{4}-\d{2}-\d{2}$", data):
        errors.append(FieldError(path=path or "/", message="Invalid date format (expected YYYY-MM-DD)"))
    elif fmt == "date-time" and not re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", data):
        errors.append(FieldError(path=path or "/", message="Invalid date-time format"))
    elif fmt == "uuid" and not re.match(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", data, re.I):
        errors.append(FieldError(path=path or "/", message="Invalid UUID format"))

    return errors


def _validate_number(data, schema: Dict[str, Any], path: str) -> List[FieldError]:
    errors = []
    if "minimum" in schema and data < schema["minimum"]:
        errors.append(FieldError(path=path or "/", message=f"Value {data} below minimum {schema['minimum']}"))
    if "maximum" in schema and data > schema["maximum"]:
        errors.append(FieldError(path=path or "/", message=f"Value {data} exceeds maximum {schema['maximum']}"))
    if "exclusiveMinimum" in schema and data <= schema["exclusiveMinimum"]:
        errors.append(FieldError(path=path or "/", message=f"Value {data} must be > {schema['exclusiveMinimum']}"))
    if "exclusiveMaximum" in schema and data >= schema["exclusiveMaximum"]:
        errors.append(FieldError(path=path or "/", message=f"Value {data} must be < {schema['exclusiveMaximum']}"))
    if "multipleOf" in schema and data % schema["multipleOf"] != 0:
        errors.append(FieldError(path=path or "/", message=f"Value {data} is not a multiple of {schema['multipleOf']}"))
    return errors
