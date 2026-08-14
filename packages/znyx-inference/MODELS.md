# Models, licenses, and the fetch-and-pin workflow

`znyx-inference` ships **code only. No model weights are bundled** in this repo or
in the Docker image. You fetch the models you want explicitly, verify them by
sha256, and mount them. Nothing is downloaded implicitly on the serving path.

This file lists the curated models, their licenses, and flags the ones that carry
special terms. **Model licenses are the operator's responsibility.** Confirm the
license of any model you deploy for your use case.

## Fetch-and-pin workflow

Export runs **offline** (the heavy `[export]` extra: torch + optimum); serving is lean
(`[onnx]`: onnxruntime + tokenizers). The torch/CUDA payload never ships in the image.

1. Export + int8-quantize a vetted checkpoint to a pinned ONNX artifact and print its sha256:
   `python -m scripts.fetch_inference_model --task prompt_injection` (operator-run, the only
   step that touches the network; needs `pip install 'znyx-inference[export]'`). It writes
   `model.onnx` (+ `model_quantized.onnx`) and `tokenizer.json` into the artifact dir.
   (Generative `guard_llm` models are snapshotted as raw weights instead of exported, and
   serve only under the `[torch]` extra.)
2. Mount the local artifact read-only and wire the task (model_id, revision, sha256) via env / compose.
3. Run the sidecar with `[onnx]` installed. It loads local, sha256-verified ONNX only, and fails closed.

## Default models (what a task pins out of the box)

| Task | Model | License | Commercial self-host | Notes |
|------|-------|---------|----------------------|-------|
| prompt_injection | protectai/deberta-v3-base-prompt-injection-v2 | Apache-2.0 | ✅ yes | OSI-open |
| toxicity | unitary/toxic-bert | Apache-2.0 | ✅ yes | OSI-open |
| language | papluca/xlm-roberta-base-language-detection | MIT | ✅ yes | OSI-open |
| nli | cross-encoder/nli-deberta-v3-large | Apache-2.0 | ✅ yes | OSI-open |
| topic_intent | sentence-transformers/all-MiniLM-L6-v2 | Apache-2.0 | ✅ yes | OSI-open |
| pii_ner | Davlan/bert-base-multilingual-cased-ner-hrl | Apache-2.0 | ✅ yes | OSI-open; 10 languages, PER/ORG/LOC |
| **safety** | **meta-llama/Llama-Guard-3-1B** | **Llama 3.2 Community** | ⚠️ conditional | **Special license - see flags below** |

## Flagged models (special licenses)

None of the curated models are non-commercial. The models below carry **special
terms** and are **not OSI-open**. They are offered as choices, but read the license
before deploying, and prefer the permissive alternatives where possible.

| Model | License | Why flagged |
|-------|---------|-------------|
| meta-llama/Llama-Guard-3-1B (default for `safety`) | Llama 3.2 Community | Commercial use allowed **only under 700M monthly active users**; carries an Acceptable Use Policy; requires "Built with Llama" attribution; the model is **gated on Hugging Face** (you must accept Meta's terms to download); not OSI-open. |
| meta-llama/Llama-Guard-3-8B (candidate: safety / guard_llm) | Llama Community | Same Llama Community terms as above. |
| meta-llama/Llama-Prompt-Guard-2-86M (candidate: prompt_injection) | Llama Community | Same Llama Community terms as above. |

### Recommendation

The only **default** with a special license is `safety` (Llama-Guard-3-1B). Every
other default is Apache-2.0 or MIT. If you want a 100% permissive out-of-box setup,
pin an Apache-2.0 guard model for `safety` instead, for example
`ibm-granite/granite-guardian-3.0-2b` (Apache-2.0, CPU-ok) or `allenai/wildguard`
(Apache-2.0, GPU). Any swapped model must clear your scorecard/benchmark gate
before it can enforce.

## Minor license notes

- **OpenRAIL++** models (some toxicity alternatives) permit commercial use but carry
  behavioral use restrictions. Read the acceptable-use terms.
- **CC-BY-SA-3.0** (fastText language id alternative) permits commercial use but is
  share-alike and requires attribution if you redistribute the model.

## Full candidate shortlist

The complete per-task shortlist (primary + alternatives, with license, size, and
hardware notes) lives in code at
`packages/znyx-core/src/znyx_core/engine/ml_catalog.py` (`CANDIDATE_MODELS`), and
is queryable via `candidate_models(open_only=True)` to drop every non-OSI row.
