"""
Tool Governance

Implements:
- Tool allowlist/denylist
- Tool argument validation (lightweight schema)
- Domain allowlist for URLs in arguments
- Exfiltration detection in tool arguments
"""
import re
import json
from typing import List, Dict, Any, Optional, Set
from urllib.parse import urlparse
from app.shared.core.models import DetectorResult, RuleHit, Severity, Decision


class ToolGovernanceDetector:
    """Validates tool invocations against governance policies"""

    # URL pattern
    URL_PATTERN = re.compile(
        r'https?://[^\s<>"{}|\\^`\[\]]+',
        re.IGNORECASE
    )

    # Exfiltration patterns in tool args
    EXFIL_PATTERNS = [
        r'\b(all|entire|complete|full)\s+(chat|conversation|history|memory|messages?)',
        r'\b(system|developer|hidden)\s+(prompt|instructions?)',
        r'\b(dump|export|reveal)\s+(all|entire|complete)',
    ]

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize tool governance detector.

        Args:
            config: Configuration dict with keys:
                - enabled: bool (default: True)
                - allowed: list of allowed tool names (if set, only these allowed)
                - denied: list of denied tool names
                - schemas: dict mapping tool_name -> schema
                - domain_allowlist: list of allowed domains for URLs
                - max_arg_size: int (default: 10000) - max size of any arg value
        """
        self.config = config
        self.enabled = config.get('enabled', True)
        # Accept both 'allowed'/'allowed_tools' and 'denied'/'blocked_tools' as aliases
        self.allowed_tools = set(config.get('allowed', []) or config.get('allowed_tools', []))
        self.denied_tools = set(config.get('denied', []) or config.get('blocked_tools', []))
        self.schemas = config.get('schemas', {})
        self.domain_allowlist = set(config.get('domain_allowlist', []))
        self.max_arg_size = config.get('max_arg_size', 10000)

        # Compile exfil patterns
        self.exfil_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.EXFIL_PATTERNS
        ]

    def _check_allowlist_denylist(self, tool_name: str) -> Optional[RuleHit]:
        """
        Check if tool is in allowlist/denylist.

        Args:
            tool_name: Name of the tool

        Returns:
            RuleHit if blocked, None if allowed
        """
        # If tool_name is a plain identifier (e.g. "send_email"), apply exact
        # match against both lists.  If it is a natural-language sentence
        # (contains whitespace), scan for denied tool names as whole words and
        # skip the allowlist check — we cannot reliably extract the tool name
        # from free text, so we only block when a denied tool is explicitly
        # mentioned.
        if re.search(r'\s', tool_name):
            for denied in self.denied_tools:
                if re.search(r'\b' + re.escape(denied) + r'\b', tool_name):
                    return RuleHit(
                        rule_id="tools.denied",
                        severity=Severity.HIGH,
                        message=f"Tool '{denied}' is on the deny list"
                    )
            return None

        # Exact-match path (plain identifier)
        # Check denylist first
        if tool_name in self.denied_tools:
            return RuleHit(
                rule_id="tools.denied",
                severity=Severity.HIGH,
                message=f"Tool '{tool_name}' is on the deny list"
            )

        # Check allowlist (if set)
        if self.allowed_tools and tool_name not in self.allowed_tools:
            return RuleHit(
                rule_id="tools.not_allowed",
                severity=Severity.HIGH,
                message=f"Tool '{tool_name}' is not in the allow list"
            )

        return None

    def _validate_args_schema(self, tool_name: str, tool_args: Dict[str, Any]) -> Optional[RuleHit]:
        """
        Validate tool arguments against lightweight schema.

        Args:
            tool_name: Name of the tool
            tool_args: Tool arguments

        Returns:
            RuleHit if validation fails, None if valid
        """
        if tool_name not in self.schemas:
            return None

        schema = self.schemas[tool_name]

        # Check required keys
        required = schema.get('required', [])
        for key in required:
            if key not in tool_args:
                return RuleHit(
                    rule_id="tools.args_invalid",
                    severity=Severity.MEDIUM,
                    message=f"Missing required argument: '{key}' for tool '{tool_name}'"
                )

        # Check types and constraints
        properties = schema.get('properties', {})
        for key, constraints in properties.items():
            if key not in tool_args:
                continue

            value = tool_args[key]
            expected_type = constraints.get('type')

            # Type validation
            type_map = {
                'string': str,
                'number': (int, float),
                'integer': int,
                'boolean': bool,
                'object': dict,
                'array': list
            }

            python_type = type_map.get(expected_type)
            if python_type and not isinstance(value, python_type):
                return RuleHit(
                    rule_id="tools.args_invalid",
                    severity=Severity.MEDIUM,
                    message=f"Argument '{key}': expected {expected_type}, got {type(value).__name__}"
                )

            # String constraints
            if expected_type == 'string':
                if 'maxLength' in constraints and len(value) > constraints['maxLength']:
                    return RuleHit(
                        rule_id="tools.args_invalid",
                        severity=Severity.MEDIUM,
                        message=f"Argument '{key}' exceeds maxLength {constraints['maxLength']}"
                    )

        # Check for unknown keys if additionalProperties=false
        if schema.get('additionalProperties') == False:
            known_keys = set(properties.keys())
            unknown_keys = set(tool_args.keys()) - known_keys
            if unknown_keys:
                return RuleHit(
                    rule_id="tools.args_invalid",
                    severity=Severity.LOW,
                    message=f"Unknown arguments: {', '.join(unknown_keys)}"
                )

        return None

    def _extract_urls(self, data: Any, urls: Set[str]):
        """
        Recursively extract URLs from nested data structure.

        Args:
            data: Data to search (dict, list, or string)
            urls: Set to accumulate found URLs
        """
        if isinstance(data, str):
            # Find URLs in string
            matches = self.URL_PATTERN.findall(data)
            urls.update(matches)
        elif isinstance(data, dict):
            for value in data.values():
                self._extract_urls(value, urls)
        elif isinstance(data, list):
            for item in data:
                self._extract_urls(item, urls)

    def _check_domain_allowlist(self, tool_args: Dict[str, Any]) -> Optional[RuleHit]:
        """
        Check if URLs in tool args are from allowed domains.

        Args:
            tool_args: Tool arguments

        Returns:
            RuleHit if blocked domain found, None if OK
        """
        if not self.domain_allowlist:
            return None

        # Extract all URLs from args
        urls: Set[str] = set()
        self._extract_urls(tool_args, urls)

        # Check each URL's domain
        for url in urls:
            try:
                parsed = urlparse(url)
                hostname = parsed.hostname
                if not hostname:
                    continue

                # Check if hostname or parent domain is in allowlist
                allowed = False
                for domain in self.domain_allowlist:
                    if hostname == domain or hostname.endswith('.' + domain):
                        allowed = True
                        break

                if not allowed:
                    return RuleHit(
                        rule_id="tools.domain_blocked",
                        severity=Severity.HIGH,
                        message=f"Domain '{hostname}' not in allowlist"
                    )

            except Exception:
                # Invalid URL, skip
                continue

        return None

    def _check_exfil_in_args(self, tool_args: Dict[str, Any]) -> Optional[RuleHit]:
        """
        Check for exfiltration patterns in tool arguments.

        Args:
            tool_args: Tool arguments

        Returns:
            RuleHit if exfil pattern detected, None if OK
        """
        # Convert args to string for pattern matching
        args_str = json.dumps(tool_args).lower()

        # Check patterns
        for pattern in self.exfil_patterns:
            if pattern.search(args_str):
                return RuleHit(
                    rule_id="tools.exfil_attempt",
                    severity=Severity.HIGH,
                    message="Exfiltration pattern detected in tool arguments"
                )

        # Check for very large payloads (potential data exfil)
        for key, value in tool_args.items():
            if isinstance(value, str) and len(value) > self.max_arg_size:
                return RuleHit(
                    rule_id="tools.exfil_attempt",
                    severity=Severity.MEDIUM,
                    message=f"Argument '{key}' is suspiciously large ({len(value)} chars)"
                )

        return None

    def detect(self, tool_name: str, tool_args: Dict[str, Any]) -> DetectorResult:
        """
        Validate tool invocation against governance policies.

        Args:
            tool_name: Name of the tool
            tool_args: Tool arguments

        Returns:
            DetectorResult with BLOCK if governance violated
        """
        if not self.enabled:
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)

        rule_hits: List[RuleHit] = []

        # 1. Check allowlist/denylist
        check = self._check_allowlist_denylist(tool_name)
        if check:
            rule_hits.append(check)

        # 2. Validate args schema
        if not rule_hits:  # Only if not already blocked
            check = self._validate_args_schema(tool_name, tool_args)
            if check:
                rule_hits.append(check)

        # 3. Check domain allowlist
        if not rule_hits:
            check = self._check_domain_allowlist(tool_args)
            if check:
                rule_hits.append(check)

        # 4. Check for exfil patterns
        if not rule_hits:
            check = self._check_exfil_in_args(tool_args)
            if check:
                rule_hits.append(check)

        # No violations
        if not rule_hits:
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)

        # Calculate risk score
        risk_score = min(len(rule_hits) * 40, 100)

        # Block tool invocation
        return DetectorResult(
            decision=Decision.BLOCK,
            risk_score=risk_score,
            rule_hits=rule_hits,
            user_message="Tool invocation blocked by governance policy.",
            developer_message=f"Tool '{tool_name}' blocked: {', '.join([hit.rule_id for hit in rule_hits])}"
        )
