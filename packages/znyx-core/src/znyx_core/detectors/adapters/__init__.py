"""Remote-API vendor moderation/safety adapters.

Each adapter maps a vendor's response onto the ``DetectorResult`` contract. A
``remote_api`` detector backend selects one via its ``provider`` field (e.g.
``openai_moderation``). The escalation path applies the egress gate + audit
before invoking the adapter; the adapter only builds the request, posts, and maps.

Implemented vendors: OpenAI Moderation, Azure AI Content Safety, and AWS Bedrock
Guardrails (SigV4-signed via the stdlib — no boto3)."""
from __future__ import annotations

from typing import Dict, List, Optional

from .aws_bedrock_guardrails import AwsBedrockGuardrailsAdapter
from .azure_content_safety import AzureContentSafetyAdapter
from .openai_moderation import OpenAIModerationAdapter

# An OpenAI-compatible moderation endpoint (self-hosted / proxy) speaks the same
# response shape, so it reuses the OpenAI adapter with a custom endpoint_url.
_ADAPTERS: Dict[str, object] = {
    "openai_moderation": OpenAIModerationAdapter(),
    "openai_compatible": OpenAIModerationAdapter(),
    "azure_content_safety": AzureContentSafetyAdapter(),
    "aws_bedrock_guardrails": AwsBedrockGuardrailsAdapter(),
}


def get_adapter(provider: Optional[str]):
    """Return the moderation adapter for a provider key, or None if unknown."""
    return _ADAPTERS.get((provider or "").strip().lower())


def adapter_names() -> List[str]:
    return sorted(_ADAPTERS)
