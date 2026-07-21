"""Optional local guard-LLM runner — a Llama-Guard-style causal LM that classifies
content as safe/unsafe. This is the ONE runner that still needs eager PyTorch (generative
decoding), so it is the opt-in escape hatch: it serves only when the ``[torch]`` extra is
installed (``pip install znyx-inference[torch]``). Under the default lean ``[onnx]`` image
torch is absent, so this task degrades to unavailable (503) instead of loading — the
CPU-ONNX classifier/nli runners cover the common guard cases. Deps and weights load only in
``load()`` from the verified local artifact dir (never the network); blocks the local_llm mode.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

from znyx_inference.runners._heavy import HeavyRunner, require
from znyx_inference.runners.base import InferOutput

_UNSAFE = re.compile(r"\bunsafe\b", re.IGNORECASE)
_SAFE = re.compile(r"\bsafe\b", re.IGNORECASE)


class GuardLlmRunner(HeavyRunner):
    runner_kind = "guard_llm"

    def _import_stack(self) -> None:
        # Needs the [torch] extra; absent under the lean [onnx] image → RunnerUnavailable.
        self._torch = require(lambda: __import__("torch"), "torch ([torch] extra)")
        require(lambda: __import__("transformers"), "transformers ([torch] extra)")

    def _build(self, path: str) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self._tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self._model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True)
        self._model.eval()
        self._max_new = int(self.spec.get("max_new_tokens", 16))
        # Optional prompt template with a {content} slot (Llama-Guard chat format, etc.).
        self._template = self.spec.get("prompt_template", "{content}")

    def _verdict(self, generated: str) -> float:
        # First explicit safe/unsafe token wins; unknown → mid risk so it surfaces.
        u = _UNSAFE.search(generated)
        s = _SAFE.search(generated)
        if u and (not s or u.start() < s.start()):
            return 1.0
        if s:
            return 0.0
        return 0.5

    def infer_batch(self, texts: List[str], params: dict | None = None) -> List[InferOutput]:
        torch = self._torch
        outs = []
        # Guard LLMs aren't reliably batchable across prompt lengths → score sequentially
        # (the BatchProcessor still coalesces the queue; one model call per item here).
        for text in texts:
            prompt = self._template.format(content=text)
            enc = self._tok(prompt, return_tensors="pt", truncation=True, max_length=4096)
            with torch.no_grad():
                gen = self._model.generate(**enc, max_new_tokens=self._max_new, do_sample=False)
            decoded = self._tok.decode(gen[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            prob = self._verdict(decoded)
            outs.append(self._output(prob, {"unsafe": prob, "safe": 1.0 - prob}))
        return outs


def make_runner(task: str, spec: Dict[str, Any]) -> GuardLlmRunner:
    return GuardLlmRunner(task, spec)
