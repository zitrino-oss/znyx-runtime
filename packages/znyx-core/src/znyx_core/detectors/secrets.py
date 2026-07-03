"""
Secrets & Credential Leakage Detector

Detects and HARD BLOCKS:
- API keys (OpenAI, Anthropic, Azure, Google)
- JWTs (JSON Web Tokens)
- Private keys (PEM format)
- Connection strings (database, cloud services)
- AWS credentials

This detector takes precedence over all others - detected secrets always result in BLOCK.
"""
import re
import base64
from typing import List, Dict, Any
from znyx_core.core.models import DetectorResult, RuleHit, Severity, Decision


class SecretsDetector:
    """Detects secrets and credentials in text with HARD BLOCK enforcement"""

    # Compiled patterns for performance
    PATTERNS = {
        # ---- Model / LLM vendor keys ----
        # OpenAI keys: sk-... or sk-proj-... (48+ chars, allows hyphens)
        'openai_key': re.compile(r'\bsk-[A-Za-z0-9_-]{20,}\b'),
        # Anthropic keys: sk-ant-...
        'anthropic_key': re.compile(r'\bsk-ant-[A-Za-z0-9_-]{20,}\b'),
        # Azure keys (32-char hex)
        'azure_key': re.compile(r'\b[A-Fa-f0-9]{32}\b'),
        # Google API keys (39 chars starting with AIza)
        'google_api_key': re.compile(r'\bAIza[A-Za-z0-9_-]{35}\b'),
        # OpenRouter: sk-or-...
        'openrouter_key': re.compile(r'\bsk-or-(?:v1-)?[A-Za-z0-9_-]{32,}\b'),
        # Hugging Face: hf_...
        'huggingface_token': re.compile(r'\bhf_[A-Za-z0-9]{30,}\b'),
        # xAI: xai-...
        'xai_key': re.compile(r'\bxai-[A-Za-z0-9]{60,}\b'),
        # Mistral API keys (32-char alnum preceded by "mistral" context or standalone bearer)
        'mistral_key': re.compile(r'\b(?:mistral[_-]?(?:api[_-]?)?key|MISTRAL_API_KEY)["\x27\s:=]+([A-Za-z0-9]{32})\b', re.IGNORECASE),
        # Cohere API keys (40-char alnum with context)
        'cohere_key': re.compile(r'\b(?:cohere[_-]?(?:api[_-]?)?key|COHERE_API_KEY)["\x27\s:=]+([A-Za-z0-9]{40})\b', re.IGNORECASE),
        # Replicate: r8_...
        'replicate_token': re.compile(r'\br8_[A-Za-z0-9]{37,}\b'),
        # DeepSeek API keys (sk-prefix, deepseek context)
        'deepseek_key': re.compile(r'\b(?:deepseek[_-]?(?:api[_-]?)?key|DEEPSEEK_API_KEY)["\x27\s:=]+([A-Za-z0-9]{32,})\b', re.IGNORECASE),
        # Groq: gsk_...
        'groq_key': re.compile(r'\bgsk_[A-Za-z0-9]{48,}\b'),

        # ---- Source control / issue trackers ----
        # GitHub classic PAT — relaxed from exactly {36} to {30,} to cover
        # shorter historical tokens (GitHub docs have stated 40 chars total
        # including prefix, but real-world tokens vary).
        'github_pat_classic': re.compile(r'\bghp_[A-Za-z0-9]{30,}\b'),
        # GitHub OAuth
        'github_oauth': re.compile(r'\bgho_[A-Za-z0-9]{30,}\b'),
        # GitHub fine-grained PAT
        'github_pat_fine': re.compile(r'\bgithub_pat_[A-Za-z0-9_]{22,}\b'),
        # GitHub App installation token
        'github_app_token': re.compile(r'\b(?:ghs_|ghu_)[A-Za-z0-9]{30,}\b'),
        # NPM access token (new — previously missing)
        'npm_token': re.compile(r'\bnpm_[A-Za-z0-9]{30,}\b'),
        # GitLab personal access token
        'gitlab_pat': re.compile(r'\bglpat-[A-Za-z0-9_-]{20}\b'),
        # GitLab runner token
        'gitlab_runner': re.compile(r'\bglrt-[A-Za-z0-9_-]{20}\b'),
        # Linear API key
        'linear_key': re.compile(r'\blin_(?:api|oauth)_[A-Za-z0-9]{40,}\b'),
        # Notion integration token
        'notion_token': re.compile(r'\bntn_[A-Za-z0-9]{40,}\b|\bsecret_[A-Za-z0-9]{43}\b'),

        # ---- Payments / email / comms ----
        # Stripe live/test/restricted/publishable
        'stripe_secret': re.compile(r'\bsk_(?:live|test)_[A-Za-z0-9]{24,}\b'),
        'stripe_restricted': re.compile(r'\brk_(?:live|test)_[A-Za-z0-9]{24,}\b'),
        'stripe_publishable': re.compile(r'\bpk_(?:live|test)_[A-Za-z0-9]{24,}\b'),
        # Twilio
        'twilio_account_sid': re.compile(r'\bAC[a-f0-9]{32}\b'),
        # Twilio API Key — broadened from lowercase-hex-32 to mixed-case
        # alnum 28-34 so real-world variants aren't missed.
        'twilio_api_key': re.compile(r'\bSK[A-Za-z0-9]{28,34}\b'),
        'twilio_auth_token': re.compile(r'\b(?:twilio[_-]?auth[_-]?token|TWILIO_AUTH_TOKEN)["\x27\s:=]+([a-f0-9]{32})\b', re.IGNORECASE),
        # SendGrid — relaxed second-segment length from {40,} to {35,} (tokens
        # in the wild are shorter than the original spec).
        'sendgrid_key': re.compile(r'\bSG\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{35,}\b'),
        # Mailgun (private key)
        'mailgun_key': re.compile(r'\bkey-[A-Za-z0-9]{32}\b'),
        # Mailchimp (hyphen + dc suffix)
        'mailchimp_key': re.compile(r'\b[0-9a-f]{32}-us[0-9]{1,2}\b'),
        # PagerDuty
        'pagerduty_token': re.compile(r'\b(?:pagerduty[_-]?(?:api[_-]?)?(?:token|key)|PAGERDUTY_API_TOKEN)["\x27\s:=]+([A-Za-z0-9+/=_-]{20,})\b', re.IGNORECASE),
        # Slack bot / user / admin / refresh
        'slack_bot_token': re.compile(r'\bxoxb-\d{9,}-\d{9,}-[A-Za-z0-9]{24,}\b'),
        'slack_user_token': re.compile(r'\bxoxp-\d{9,}-\d{9,}-\d{9,}-[A-Za-z0-9]{32,}\b'),
        'slack_admin_token': re.compile(r'\bxoxa-[A-Za-z0-9-]{40,}\b'),
        'slack_refresh_token': re.compile(r'\bxoxr-[A-Za-z0-9-]{40,}\b'),
        'slack_webhook': re.compile(r'\bhttps://hooks\.slack\.com/services/T[A-Z0-9]{8,}/B[A-Z0-9]{8,}/[A-Za-z0-9]{24}\b'),

        # ---- Observability / infra / vault ----
        # Datadog API + APP keys (context-guarded for hex patterns)
        'datadog_api_key': re.compile(r'\b(?:datadog[_-]?api[_-]?key|DD_API_KEY|DATADOG_API_KEY)["\x27\s:=]+([a-f0-9]{32})\b', re.IGNORECASE),
        'datadog_app_key': re.compile(r'\b(?:datadog[_-]?(?:app|application)[_-]?key|DD_APP_KEY|DATADOG_APP_KEY)["\x27\s:=]+([a-f0-9]{40})\b', re.IGNORECASE),
        # HashiCorp Vault
        'hashicorp_vault': re.compile(r'\bhvs\.[A-Za-z0-9_-]{24,}\b|\bhvb\.[A-Za-z0-9_-]{24,}\b'),
        # Doppler (service, config, personal, service-account tokens with optional env segment)
        'doppler_token': re.compile(r'\bdp\.(?:st|ct|pt|sa)\.(?:[A-Za-z0-9_-]+\.)?[A-Za-z0-9_-]{40,}\b'),
        # Vercel
        'vercel_token': re.compile(r'\b(?:vercel[_-]?(?:api[_-]?)?token|VERCEL_TOKEN)["\x27\s:=]+([A-Za-z0-9]{24})\b', re.IGNORECASE),
        # Heroku
        'heroku_api_key': re.compile(r'\b(?:heroku[_-]?api[_-]?key|HEROKU_API_KEY)["\x27\s:=]+([a-f0-9-]{36})\b', re.IGNORECASE),
        # Cloudflare
        'cloudflare_api_token': re.compile(r'\b(?:cloudflare[_-]?(?:api[_-]?)?token|CF_API_TOKEN)["\x27\s:=]+([A-Za-z0-9_-]{40})\b', re.IGNORECASE),
        # Generic API key patterns (fallback, keyword-anchored)
        'generic_api_key': re.compile(r'\b(?:api[_-]?key|apikey|access[_-]?key)["\x27\s:=]+([A-Za-z0-9_-]{20,})\b', re.IGNORECASE),

        # ---- Tokens / keys / secrets ----
        # JWT tokens (three base64url segments). No trailing \b — the signature
        # can end on `=`, `+`, or `/` (standard base64 padding chars) which
        # are non-word, causing \b to refuse to match. Signature class widened
        # accordingly.
        'jwt': re.compile(r'\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_+/=-]+'),
        # Generic PEM private key (all algorithms)
        'private_key': re.compile(r'-----BEGIN[A-Z\s]+PRIVATE KEY-----'),
        # SSH key formats (OpenSSH + RSA-specific public-block leading text)
        'ssh_openssh_private': re.compile(r'-----BEGIN OPENSSH PRIVATE KEY-----'),
        'ssh_pkcs8_private': re.compile(r'-----BEGIN (?:ENCRYPTED )?PRIVATE KEY-----'),
        'ssh_ecdsa_private': re.compile(r'-----BEGIN EC PRIVATE KEY-----'),
        'ssh_ed25519_private': re.compile(r'-----BEGIN OPENSSH PRIVATE KEY-----[\s\S]{20,}ssh-ed25519'),

        # ---- Databases ----
        'postgres_conn': re.compile(r'postgres(?:ql)?://[^\s]+', re.IGNORECASE),
        'mongodb_conn': re.compile(r'mongodb(?:\+srv)?://[^\s]+', re.IGNORECASE),
        'mysql_conn': re.compile(r'mysql://[^\s]+', re.IGNORECASE),

        # ---- Cloud (Azure / AWS / GCP) ----
        # Azure connection strings
        'azure_conn': re.compile(r'(?:AccountKey|SharedAccessKey|SharedAccessSignature)=[A-Za-z0-9+/=]{20,}', re.IGNORECASE),
        # Azure SAS token URL parameter
        'azure_sas': re.compile(r'\?sv=\d{4}-\d{2}-\d{2}&[^\s]*?sig=[A-Za-z0-9%]{40,}'),
        # Azure Storage account key (88-char base64, context-anchored)
        'azure_storage_key': re.compile(r'\b(?:AccountKey|StorageAccountKey|AZURE_STORAGE_KEY)["\x27\s:=]+([A-Za-z0-9+/]{86}==)', re.IGNORECASE),
        # AWS credentials
        'aws_access_key': re.compile(r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'),
        'aws_secret_key': re.compile(r'\b(?:aws_secret_access_key|AWS_SECRET_ACCESS_KEY)["\x27\s:=]+([A-Za-z0-9+/]{40})\b'),
        # AWS session token (long base64)
        'aws_session_token': re.compile(r'\b(?:aws_session_token|AWS_SESSION_TOKEN)["\x27\s:=]+([A-Za-z0-9+/=]{100,})'),
        # GCP service account JSON (detect the full private_key + type marker together)
        'gcp_service_account': re.compile(r'"type"\s*:\s*"service_account"[\s\S]*?"private_key"\s*:\s*"-----BEGIN[^"]*?PRIVATE KEY-----'),
    }

    # Base64-encoded secrets: look for base64 strings (24+ chars) that decode to known key prefixes
    BASE64_PATTERN = re.compile(r'[A-Za-z0-9+/]{24,}={0,2}')

    # Known key prefixes that might appear inside decoded base64
    DECODED_KEY_PREFIXES = [
        b'sk-', b'sk-ant-', b'AIza', b'AKIA', b'ASIA',
        b'-----BEGIN', b'postgres://', b'mongodb://', b'mysql://',
    ]

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize secrets detector.

        Args:
            config: Configuration dict with keys:
                - enabled: bool (default: True)
                - action: "BLOCK" or "REDACT" (default: "BLOCK")
                - exceptions: list of patterns to ignore (optional)
                - detect_base64: bool - scan for base64-encoded secrets (default: True)
        """
        self.config = config
        self.enabled = config.get('enabled', True)
        self.action = config.get('action', 'BLOCK').upper()
        self.exceptions = config.get('exceptions', [])
        self.detect_base64 = config.get('detect_base64', True)

    def _is_likely_jwt(self, token: str) -> bool:
        """
        Validate if a token is likely a JWT by checking structure.

        Args:
            token: Potential JWT token

        Returns:
            True if likely a JWT
        """
        parts = token.split('.')
        if len(parts) != 3:
            return False

        # Check if parts are valid base64url (header and payload should decode)
        try:
            # Try to decode header and payload
            base64.urlsafe_b64decode(parts[0] + '==')  # Add padding
            base64.urlsafe_b64decode(parts[1] + '==')
            return True
        except Exception:
            return False

    def _check_entropy(self, text: str, threshold: float = 3.5) -> bool:
        """
        Shannon entropy check for high-entropy strings (likely secrets).

        Args:
            text: Text to check
            threshold: Entropy threshold in bits per char (default 3.5 -
                realistic 32-hex-char Azure keys come in around 3.8 bits
                due to finite-sample variation, 3.5 admits them while still
                rejecting patterned content like "deadbeef…" repeats).

        Returns:
            True if entropy is high (likely random/secret), False if patterned.
        """
        import math

        if len(text) < 20:
            return False

        freq: Dict[str, int] = {}
        for char in text:
            freq[char] = freq.get(char, 0) + 1

        # Proper Shannon entropy: H = -Σ p_i * log2(p_i)
        length = len(text)
        entropy = 0.0
        for count in freq.values():
            p = count / length
            if p > 0:
                entropy -= p * math.log2(p)

        # Use >= so perfect-uniform hex strings (entropy exactly 4.0) pass.
        return entropy >= threshold

    def _detect_base64_secrets(self, text: str) -> List[Dict[str, Any]]:
        """
        Scan for base64-encoded strings that decode to known secret patterns.

        Returns list of dicts with keys: rule_name, matched_text, start, end
        """
        findings: List[Dict[str, Any]] = []
        if not self.detect_base64:
            return findings

        for match in self.BASE64_PATTERN.finditer(text):
            b64_str = match.group(0)

            # Skip if it looks like a JWT segment (handled separately)
            if '.' in text[max(0, match.start()-1):match.end()+1]:
                continue

            try:
                decoded = base64.b64decode(b64_str, validate=True)
            except Exception:
                continue

            # Check if decoded content contains known key prefixes
            for prefix in self.DECODED_KEY_PREFIXES:
                if prefix in decoded:
                    findings.append({
                        'rule_name': 'base64_encoded_secret',
                        'matched_text': b64_str,
                        'start': match.start(),
                        'end': match.end(),
                        'decoded_hint': prefix.decode('utf-8', errors='replace'),
                    })
                    break
            else:
                # No known prefix - check entropy of the decoded bytes
                if len(decoded) >= 20 and self._check_entropy(b64_str):
                    # Also run the plaintext patterns against the decoded content
                    decoded_text = decoded.decode('utf-8', errors='replace')
                    for rule_name, pattern in self.PATTERNS.items():
                        if pattern.search(decoded_text):
                            findings.append({
                                'rule_name': f'base64_{rule_name}',
                                'matched_text': b64_str,
                                'start': match.start(),
                                'end': match.end(),
                                'decoded_hint': rule_name,
                            })
                            break

        return findings

    def _redact_text(self, text: str, secrets: List[Dict[str, Any]]) -> str:
        """Replace each detected secret in text with a [REDACTED] placeholder."""
        # Sort by position descending so replacements don't shift offsets
        sorted_secrets = sorted(secrets, key=lambda s: s['start'], reverse=True)
        result = text
        for secret in sorted_secrets:
            result = result[:secret['start']] + '[REDACTED]' + result[secret['end']:]
        return result

    def detect(self, text: str) -> DetectorResult:
        """
        Detect secrets and credentials in text.

        Args:
            text: Input text to scan

        Returns:
            DetectorResult with BLOCK or REDACT decision if secrets found
        """
        if not self.enabled:
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)

        rule_hits: List[RuleHit] = []
        # Track positions for redaction
        secret_spans: List[Dict[str, Any]] = []

        # Check each pattern
        for rule_name, pattern in self.PATTERNS.items():
            matches = pattern.finditer(text)

            for match in matches:
                matched_text = match.group(0)

                # Check exceptions (e.g., placeholder values)
                if any(exc in matched_text for exc in self.exceptions):
                    continue

                # Additional validation for specific types
                if rule_name == 'jwt':
                    if not self._is_likely_jwt(matched_text):
                        continue

                if rule_name == 'azure_key':
                    # Azure keys should have high entropy
                    if not self._check_entropy(matched_text):
                        continue

                # Mask the secret in the message (show first 4 and last 4 chars)
                if len(matched_text) > 12:
                    masked = f"{matched_text[:4]}...{matched_text[-4:]}"
                else:
                    masked = "***REDACTED***"

                rule_hits.append(RuleHit(
                    rule_id=f"secrets.{rule_name}",
                    severity=Severity.CRITICAL if hasattr(Severity, 'CRITICAL') else Severity.HIGH,
                    message=f"Secret detected: {rule_name} ({masked})"
                ))
                secret_spans.append({
                    'start': match.start(),
                    'end': match.end(),
                })

        # Check for base64-encoded secrets
        b64_findings = self._detect_base64_secrets(text)
        for finding in b64_findings:
            b64_str = finding['matched_text']
            if any(exc in b64_str for exc in self.exceptions):
                continue

            if len(b64_str) > 12:
                masked = f"{b64_str[:4]}...{b64_str[-4:]}"
            else:
                masked = "***REDACTED***"

            rule_hits.append(RuleHit(
                rule_id=f"secrets.{finding['rule_name']}",
                severity=Severity.CRITICAL if hasattr(Severity, 'CRITICAL') else Severity.HIGH,
                message=f"Base64-encoded secret detected: {finding['decoded_hint']} ({masked})"
            ))
            secret_spans.append({
                'start': finding['start'],
                'end': finding['end'],
            })

        # No secrets found
        if not rule_hits:
            return DetectorResult(decision=Decision.ALLOW, risk_score=0)

        # Determine action: BLOCK or REDACT
        if self.action == 'REDACT':
            sanitized = self._redact_text(text, secret_spans)
            return DetectorResult(
                decision=Decision.REDACT,
                risk_score=100,
                rule_hits=rule_hits,
                sanitized_text=sanitized,
                user_message="Sensitive credentials were detected and redacted from your request.",
                developer_message=f"CRITICAL: {len(rule_hits)} secret(s) detected - redacted"
            )

        # Default: BLOCK
        return DetectorResult(
            decision=Decision.BLOCK,
            risk_score=100,
            rule_hits=rule_hits,
            user_message="Your request contains sensitive credentials that cannot be processed.",
            developer_message=f"CRITICAL: {len(rule_hits)} secret(s) detected - request blocked"
        )
