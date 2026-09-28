"""ZNYX Inference Service — an OPTIONAL, separately-deployable sidecar that hosts
local transformer/embedding/NLI/guard-LLM models behind a stable scoring contract
spoken by the extended RemoteDetector transport.

Hard rule: the core OSS runtime and control plane gain ZERO heavy dependencies. The
heavy stack (torch/transformers/sentence-transformers/onnxruntime) lives only in this
package's runners, lazy-imported at load time, and only in the inference Docker image
(requirements-inference.txt). With no heavy deps installed the service still runs on the
dependency-free StubRunner, so the contract / cache / batching / registry are testable
anywhere.
"""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("znyx-inference")
except PackageNotFoundError:  # not installed (source-tree import)
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
