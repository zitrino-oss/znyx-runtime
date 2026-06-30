# @znyx/runtime

ZNYX Runtime binary wrapper for npm. Installs the `znyx` CLI on macOS, Linux, and Windows.

## Install

```bash
npm install -g @znyx/runtime
```

On install, a platform-matched binary is downloaded from GitHub Releases and its
SHA256 checksum is verified before being marked executable.

## Usage

```bash
znyx start                                # local mode, port 8080
znyx start --port 9090                    # custom port
znyx start --policy ./policies.yaml       # custom policy file
znyx start --mode managed                 # connect to console.znyx.ai
znyx version                              # print version
znyx init --output ./config/policies.yaml # scaffold a starter policy
```

## Supported platforms

| Platform | Binary |
|---|---|
| macOS Apple Silicon | `znyx-runtime-darwin-arm64` |
| macOS Intel | `znyx-runtime-darwin-x64` |
| Linux x64 | `znyx-runtime-linux-x64` |
| Linux ARM64 | `znyx-runtime-linux-arm64` |
| Windows x64 | `znyx-runtime-windows-x64.exe` |

On unsupported platforms, install via `pip install znyx-runtime` or Docker instead.

## Alternative installs

```bash
# Python
pip install znyx-runtime

# Docker
docker run -p 8080:8080 znyx/runtime
```

## License

Apache-2.0
