# Rapfi binaries

Store one Rapfi executable per target platform:

- `macos-arm64/rapfi` - Apple Silicon macOS build.
- `linux-aarch64/rapfi` - Raspberry Pi / Linux ARM64 build.

The Python code chooses the platform-specific binary automatically unless
`game.ai.engine_path` is set in the YAML config.
