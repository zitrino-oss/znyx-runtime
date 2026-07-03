"""
Code Safety Detector.

Scans LLM-generated code for security vulnerabilities:
SQL injection, XSS, command injection, path traversal,
insecure deserialization, SSRF, and unsafe function calls.
"""
import re
import logging
from typing import Any, Dict, List, Tuple

from znyx_core.core.models import DetectorResult, RuleHit, Severity, Decision
from znyx_core.core.risk import calculate_risk_score

logger = logging.getLogger(__name__)

# ── Code block extraction ──────────────────────────────────────────────────

_FENCED_BLOCK_RE = re.compile(
    r"```(\w*)\n(.*?)```", re.DOTALL
)
_INDENTED_BLOCK_RE = re.compile(
    r"(?:^|\n)((?:(?:    |\t).+\n?){3,})", re.MULTILINE
)


def _extract_code_blocks(text: str) -> List[Tuple[str, str]]:
    """Extract code blocks from markdown. Returns list of (language, code)."""
    blocks: List[Tuple[str, str]] = []
    for m in _FENCED_BLOCK_RE.finditer(text):
        lang = m.group(1).lower().strip()
        code = m.group(2)
        blocks.append((lang, code))
    # Also check for indented blocks if no fenced blocks found
    if not blocks:
        for m in _INDENTED_BLOCK_RE.finditer(text):
            blocks.append(("", m.group(1)))
    return blocks


def _detect_language(code: str, hint: str) -> str:
    """Detect code language from hint or heuristics."""
    if hint in ("python", "py", "python3"):
        return "python"
    if hint in ("javascript", "js", "typescript", "ts", "jsx", "tsx"):
        return "javascript"
    if hint in ("sql", "mysql", "postgresql", "postgres", "sqlite"):
        return "sql"
    if hint in ("bash", "sh", "shell", "zsh"):
        return "shell"
    if hint in ("go", "golang"):
        return "go"
    if hint in ("java",):
        return "java"
    if hint in ("php",):
        return "php"
    if hint in ("ruby", "rb"):
        return "ruby"

    # Heuristic detection
    if re.search(r"\bimport\s+\w+|from\s+\w+\s+import|def\s+\w+\s*\(", code):
        return "python"
    if re.search(r"\bfunction\s+\w+|const\s+\w+\s*=|=>\s*\{|require\(|import\s+.*from", code):
        return "javascript"
    if re.search(r"\bSELECT\b.*\bFROM\b|\bINSERT\s+INTO\b|\bUPDATE\b.*\bSET\b", code, re.I):
        return "sql"
    if re.search(r"^#!/bin/(?:ba)?sh|^\s*(?:if|for|while)\s+\[", code, re.M):
        return "shell"
    return "unknown"


# ── Vulnerability patterns per language ────────────────────────────────────
# Each entry: (regex, severity, vuln_class, message)

VulnPattern = Tuple[re.Pattern, Severity, str, str]

_PYTHON_PATTERNS: List[VulnPattern] = [
    # Command injection
    (re.compile(r"\bos\.system\s*\(", re.I), Severity.HIGH, "command_injection",
     "os.system() can execute arbitrary commands - use subprocess with shell=False"),
    (re.compile(r"\bos\.popen\s*\(", re.I), Severity.HIGH, "command_injection",
     "os.popen() can execute arbitrary commands - use subprocess with shell=False"),
    (re.compile(r"\bsubprocess\.(?:call|run|Popen)\s*\([^)]*shell\s*=\s*True", re.I),
     Severity.HIGH, "command_injection",
     "subprocess with shell=True allows shell injection - use shell=False with argument list"),
    (re.compile(r"\b(?:eval|exec)\s*\(", re.I), Severity.HIGH, "command_injection",
     "eval()/exec() executes arbitrary code - avoid with untrusted input"),
    (re.compile(r"\b__import__\s*\(", re.I), Severity.MEDIUM, "command_injection",
     "__import__() enables dynamic code loading - potential code injection vector"),

    # SQL injection
    (re.compile(r"""f["'](?:SELECT|INSERT|UPDATE|DELETE|DROP)\b[^"']*\{""", re.I),
     Severity.HIGH, "sql_injection",
     "SQL query built with f-string interpolation - use parameterized queries"),
    (re.compile(r"""["'](?:SELECT|INSERT|UPDATE|DELETE)\b[^"']{0,200}["']\s*\+\s*""", re.I),
     Severity.HIGH, "sql_injection",
     "SQL query built with string concatenation - use parameterized queries"),
    (re.compile(r"""["'](?:SELECT|INSERT|UPDATE|DELETE)\b[^"']{0,200}%s[^"']{0,200}%\s*\(""", re.I),
     Severity.MEDIUM, "sql_injection",
     "SQL query built with % formatting - use parameterized queries"),
    (re.compile(r"""\.format\s*\([^)]{0,100}\)[^;]{0,200}(?:SELECT|INSERT|UPDATE|DELETE)\b""", re.I),
     Severity.HIGH, "sql_injection",
     "SQL query built with .format() - use parameterized queries"),

    # Insecure deserialization
    (re.compile(r"\bpickle\.(?:loads?|Unpickler)\s*\(", re.I), Severity.HIGH, "insecure_deserialization",
     "pickle.loads() deserializes arbitrary objects - can execute code on untrusted data"),
    (re.compile(r"\byaml\.(?:load|unsafe_load)\s*\(", re.I), Severity.HIGH, "insecure_deserialization",
     "yaml.load() without SafeLoader can execute arbitrary code - use yaml.safe_load()"),
    (re.compile(r"\bmarshal\.loads?\s*\(", re.I), Severity.HIGH, "insecure_deserialization",
     "marshal.loads() is unsafe with untrusted data"),

    # Path traversal
    (re.compile(r"""open\s*\([^)]*(?:\.\./|\.\.\\)"""), Severity.HIGH, "path_traversal",
     "File path contains ../ - potential path traversal vulnerability"),
    (re.compile(r"""open\s*\([^)]*\+\s*(?:request|user|input|param)""", re.I),
     Severity.HIGH, "path_traversal",
     "File open with user-controlled path - validate and sanitize path input"),

    # SSRF
    (re.compile(r"\brequests\.(?:get|post|put|delete|patch)\s*\([^)]*(?:\+|\.format|f[\"'])", re.I),
     Severity.MEDIUM, "ssrf",
     "HTTP request with dynamic URL - validate against allowlist to prevent SSRF"),
    (re.compile(r"\burllib\.request\.urlopen\s*\([^)]*(?:\+|\.format|f[\"'])", re.I),
     Severity.MEDIUM, "ssrf",
     "URL open with dynamic URL - validate against allowlist to prevent SSRF"),
]

_JAVASCRIPT_PATTERNS: List[VulnPattern] = [
    # XSS
    (re.compile(r"\.innerHTML\s*=", re.I), Severity.HIGH, "xss",
     "innerHTML assignment can execute scripts - use textContent or sanitize input"),
    (re.compile(r"\.outerHTML\s*=", re.I), Severity.HIGH, "xss",
     "outerHTML assignment can execute scripts - use textContent or sanitize input"),
    (re.compile(r"\bdocument\.write\s*\(", re.I), Severity.HIGH, "xss",
     "document.write() can inject scripts - use safe DOM methods"),
    (re.compile(r"dangerouslySetInnerHTML", re.I), Severity.MEDIUM, "xss",
     "dangerouslySetInnerHTML bypasses React's XSS protection - sanitize content first"),

    # Command injection
    (re.compile(r"\beval\s*\(", re.I), Severity.HIGH, "command_injection",
     "eval() executes arbitrary code - avoid with untrusted input"),
    (re.compile(r"\bnew\s+Function\s*\(", re.I), Severity.HIGH, "command_injection",
     "new Function() creates code from strings - same risks as eval()"),
    (re.compile(r"\bsetTimeout\s*\(\s*[\"'`]", re.I), Severity.MEDIUM, "command_injection",
     "setTimeout with string argument acts like eval - use function reference"),
    (re.compile(r"\bsetInterval\s*\(\s*[\"'`]", re.I), Severity.MEDIUM, "command_injection",
     "setInterval with string argument acts like eval - use function reference"),
    (re.compile(r"\bchild_process\.exec\s*\(", re.I), Severity.HIGH, "command_injection",
     "child_process.exec() can execute shell commands - use execFile with arguments array"),

    # SQL injection
    (re.compile(r"""`(?:SELECT|INSERT|UPDATE|DELETE)\b[^`]*\$\{""", re.I),
     Severity.HIGH, "sql_injection",
     "SQL query with template literal interpolation - use parameterized queries"),
    (re.compile(r"""["'](?:SELECT|INSERT|UPDATE|DELETE)\b[^"']{0,200}["']\s*\+""", re.I),
     Severity.HIGH, "sql_injection",
     "SQL query with string concatenation - use parameterized queries"),

    # Prototype pollution
    (re.compile(r"""(?:__proto__|constructor\s*\[|prototype\s*\[)""", re.I),
     Severity.MEDIUM, "prototype_pollution",
     "Prototype pollution risk - validate property names before assignment"),
]

_SQL_PATTERNS: List[VulnPattern] = [
    # String concatenation SQL injection (Python/JS style: "SELECT ..." + user_input)
    (re.compile(r"""["'](?:SELECT|INSERT|UPDATE|DELETE)\b[^"']{0,200}["']\s*\+""", re.I),
     Severity.HIGH, "sql_injection",
     "SQL query built with string concatenation - use parameterized queries"),
    (re.compile(r"\bUNION\s+(?:ALL\s+)?SELECT\b", re.I), Severity.HIGH, "sql_injection",
     "UNION-based SQL injection pattern detected"),
    (re.compile(r";\s*(?:DROP|DELETE|TRUNCATE|ALTER|CREATE)\s+", re.I), Severity.HIGH, "sql_injection",
     "Stacked query with destructive operation - potential SQL injection"),
    (re.compile(r"--\s*$", re.M), Severity.LOW, "sql_injection",
     "SQL comment at end of line - common injection technique"),
    (re.compile(r"\bOR\s+['\"]\s*['\"]\s*=\s*['\"]\s*['\"]", re.I), Severity.HIGH, "sql_injection",
     "Classic SQL injection pattern: OR ''=''"),
    (re.compile(r"\bOR\s+1\s*=\s*1\b", re.I), Severity.HIGH, "sql_injection",
     "Classic SQL injection pattern: OR 1=1"),
    (re.compile(r"\bWAITFOR\s+DELAY\b|\bSLEEP\s*\(\d+\)", re.I), Severity.HIGH, "sql_injection",
     "Time-based SQL injection pattern detected"),
]

_SHELL_PATTERNS: List[VulnPattern] = [
    (re.compile(r"\beval\s+", re.I), Severity.HIGH, "command_injection",
     "Shell eval can execute arbitrary commands"),
    (re.compile(r"\bcurl\b.+\|\s*(?:ba)?sh\b", re.I), Severity.HIGH, "command_injection",
     "Piping curl output to shell - can execute arbitrary remote code"),
    (re.compile(r"\bwget\b.+\|\s*(?:ba)?sh\b", re.I), Severity.HIGH, "command_injection",
     "Piping wget output to shell - can execute arbitrary remote code"),
    (re.compile(r"\bchmod\s+777\b"), Severity.MEDIUM, "insecure_permissions",
     "chmod 777 gives world-writable permissions - use restrictive permissions"),
    (re.compile(r"\bchmod\s+666\b"), Severity.MEDIUM, "insecure_permissions",
     "chmod 666 gives world-readable/writable permissions"),
    (re.compile(r"""\$\{?\w+\}?(?!\")"""), Severity.LOW, "command_injection",
     "Unquoted variable expansion - may cause word splitting or glob expansion"),
    (re.compile(r"`[^`]*\$\w+[^`]*`"), Severity.MEDIUM, "command_injection",
     "Variable in backtick command substitution - quote variables to prevent injection"),
]

# Language → patterns mapping
_LANGUAGE_PATTERNS: Dict[str, List[VulnPattern]] = {
    "python": _PYTHON_PATTERNS,
    "javascript": _JAVASCRIPT_PATTERNS,
    "sql": _SQL_PATTERNS,
    "shell": _SHELL_PATTERNS,
}


class CodeSafetyDetector:
    """Scans LLM-generated code for security vulnerabilities."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get("enabled", True)
        self.action = config.get("action", "BLOCK")
        self.languages = config.get("languages", list(_LANGUAGE_PATTERNS.keys()))
        self.vuln_classes = config.get("vulnerability_classes", None)  # None = all
        self.block_unsafe_functions = config.get("block_unsafe_functions", True)

        # Custom banned functions: {"python": ["dangerous_func"], ...}
        self.custom_banned: Dict[str, List[str]] = config.get("custom_banned_functions", {})

    def _check_custom_banned(self, code: str, language: str) -> List[RuleHit]:
        """Check for custom banned function calls."""
        hits: List[RuleHit] = []
        banned = self.custom_banned.get(language, [])
        for func_name in banned:
            pattern = re.compile(rf"\b{re.escape(func_name)}\s*\(", re.I)
            for match in pattern.finditer(code):
                hits.append(RuleHit(
                    rule_id="code.custom_banned_function",
                    severity=Severity.HIGH,
                    message=f"Banned function '{func_name}' used in {language} code",
                ))
        return hits

    def _scan_code(self, code: str, language: str) -> List[RuleHit]:
        """Scan a code block for vulnerabilities."""
        hits: List[RuleHit] = []

        # Get patterns for this language
        patterns = _LANGUAGE_PATTERNS.get(language, [])

        # Also check SQL patterns in any language (SQL strings can appear anywhere)
        if language != "sql":
            patterns = patterns + _SQL_PATTERNS

        for pattern, severity, vuln_class, message in patterns:
            # Filter by requested vulnerability classes
            if self.vuln_classes is not None and vuln_class not in self.vuln_classes:
                continue

            for match in pattern.finditer(code):
                hits.append(RuleHit(
                    rule_id=f"code.{vuln_class}",
                    severity=severity,
                    message=message,
                ))

        # Custom banned functions
        if self.block_unsafe_functions:
            hits.extend(self._check_custom_banned(code, language))

        return hits

    def detect(self, text: str) -> DetectorResult:
        if not self.enabled:
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)

        # Extract code blocks; fall back to scanning raw text if none found
        blocks = _extract_code_blocks(text)
        if not blocks:
            blocks = [("", text)]

        all_hits: List[RuleHit] = []

        for hint, code in blocks:
            language = _detect_language(code, hint)

            # Skip if language not in requested list
            if language not in self.languages and language != "unknown":
                continue

            # For unknown language, try all patterns
            if language == "unknown":
                for lang in self.languages:
                    if lang in _LANGUAGE_PATTERNS:
                        all_hits.extend(self._scan_code(code, lang))
            else:
                all_hits.extend(self._scan_code(code, language))

        if not all_hits:
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)

        # Deduplicate by rule_id + message
        seen = set()
        unique_hits: List[RuleHit] = []
        for hit in all_hits:
            key = (hit.rule_id, hit.message)
            if key not in seen:
                seen.add(key)
                unique_hits.append(hit)

        # Risk score
        risk_score = calculate_risk_score(unique_hits)

        decision = Decision.BLOCK if self.action == "BLOCK" else Decision.WARN

        return DetectorResult(
            decision=decision,
            risk_score=risk_score,
            rule_hits=unique_hits,
            user_message="The generated code may contain security vulnerabilities.",
            developer_message=f"code_safety: {len(unique_hits)} vulnerabilities detected in generated code",
        )
