"""Known AI services by host, for shadow-AI discovery (AI gateway Phase 5).

``znyx discover`` reads proxy, DNS and network logs inside the customer's network and matches
each destination host against this catalogue; traffic to a listed service that did not go
through the gateway is use of AI the organization may not know about. A pattern matches the
host itself and its subdomains (``claude.ai`` matches ``claude.ai`` and ``api.claude.ai``); a
``*`` matches one or more characters within the host.

Categories: ``api`` (model APIs an app or script calls), ``chat`` (assistants people use in a
browser), ``coding`` (coding assistants), ``media`` (image, voice and video), ``models`` (model
hubs and routers). Hosts change; the list is deliberately conservative and easy to extend.
"""
from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple


@dataclass(frozen=True)
class Service:
    name: str
    vendor: str
    category: str
    hosts: Tuple[str, ...]


CATALOGUE: Tuple[Service, ...] = (
    # model APIs
    Service("OpenAI API", "OpenAI", "api", ("api.openai.com",)),
    Service("Azure OpenAI", "Microsoft", "api", ("*.openai.azure.com",)),
    Service("Azure AI services", "Microsoft", "api", ("*.cognitiveservices.azure.com", "*.services.ai.azure.com")),
    Service("Anthropic API", "Anthropic", "api", ("api.anthropic.com",)),
    Service("Gemini API", "Google", "api", ("generativelanguage.googleapis.com",)),
    Service("Vertex AI", "Google", "api", ("aiplatform.googleapis.com", "*-aiplatform.googleapis.com")),
    Service("Amazon Bedrock", "Amazon", "api", ("bedrock-runtime.*.amazonaws.com", "bedrock.*.amazonaws.com",
                                                 "bedrock-agent-runtime.*.amazonaws.com")),
    Service("Mistral API", "Mistral AI", "api", ("api.mistral.ai", "codestral.mistral.ai")),
    Service("Cohere API", "Cohere", "api", ("api.cohere.ai", "api.cohere.com")),
    Service("Groq API", "Groq", "api", ("api.groq.com",)),
    Service("Together AI", "Together", "api", ("api.together.xyz", "api.together.ai")),
    Service("Fireworks AI", "Fireworks", "api", ("api.fireworks.ai",)),
    Service("Perplexity API", "Perplexity", "api", ("api.perplexity.ai",)),
    Service("DeepSeek API", "DeepSeek", "api", ("api.deepseek.com",)),
    Service("xAI API", "xAI", "api", ("api.x.ai",)),
    Service("AI21 API", "AI21 Labs", "api", ("api.ai21.com",)),
    Service("Replicate", "Replicate", "api", ("api.replicate.com",)),
    Service("Hugging Face Inference", "Hugging Face", "api", ("api-inference.huggingface.co", "router.huggingface.co")),
    # model hubs and routers
    Service("OpenRouter", "OpenRouter", "models", ("openrouter.ai",)),
    Service("Hugging Face", "Hugging Face", "models", ("huggingface.co", "hf.co")),
    # assistants
    Service("ChatGPT", "OpenAI", "chat", ("chatgpt.com", "chat.openai.com")),
    Service("Claude", "Anthropic", "chat", ("claude.ai",)),
    Service("Gemini", "Google", "chat", ("gemini.google.com", "bard.google.com")),
    Service("Microsoft Copilot", "Microsoft", "chat", ("copilot.microsoft.com",)),
    Service("Perplexity", "Perplexity", "chat", ("perplexity.ai",)),
    Service("DeepSeek", "DeepSeek", "chat", ("chat.deepseek.com",)),
    Service("Le Chat", "Mistral AI", "chat", ("chat.mistral.ai",)),
    Service("Grok", "xAI", "chat", ("grok.com",)),
    Service("Meta AI", "Meta", "chat", ("meta.ai",)),
    Service("Poe", "Quora", "chat", ("poe.com",)),
    Service("Character.AI", "Character.AI", "chat", ("character.ai",)),
    # coding assistants
    Service("GitHub Copilot", "GitHub", "coding", ("api.githubcopilot.com", "*.githubcopilot.com",
                                                    "copilot-proxy.githubusercontent.com")),
    Service("Cursor", "Anysphere", "coding", ("cursor.sh", "cursor.com", "api2.cursor.sh")),
    Service("Windsurf", "Windsurf", "coding", ("codeium.com", "windsurf.com")),
    Service("Tabnine", "Tabnine", "coding", ("tabnine.com",)),
    # media
    Service("Midjourney", "Midjourney", "media", ("midjourney.com",)),
    Service("ElevenLabs", "ElevenLabs", "media", ("elevenlabs.io",)),
    Service("Stability AI", "Stability AI", "media", ("api.stability.ai",)),
)


def _matches(host: str, pattern: str) -> bool:
    if "*" in pattern:
        return fnmatch.fnmatchcase(host, pattern) or fnmatch.fnmatchcase(host, "*." + pattern)
    return host == pattern or host.endswith("." + pattern)


@lru_cache(maxsize=65536)
def match_host(host: Optional[str]) -> Optional[Service]:
    """The service a destination host belongs to, or None. Hosts are compared in lower case,
    without a port or a trailing dot."""
    if not host:
        return None
    host = host.strip().lower().rstrip(".")
    if host.startswith("[") or not host:
        return None
    if host.count(":") == 1:
        host = host.split(":", 1)[0]
    for service in CATALOGUE:
        if any(_matches(host, p) for p in service.hosts):
            return service
    return None
