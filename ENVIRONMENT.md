# Environment Setup for ChaosBF

This document describes the dependencies and toolchain requirements for the ChaosBF project.

## Python Dependencies

ChaosBF is primarily a Python project requiring Python 3.8 or higher.

### Installation

```bash
pip install -r requirements.txt
```

### Required Packages
- `numpy>=1.24.0` - For numerical computations
- `matplotlib>=3.7.0` - For visualization

## Rust Toolchain

The project uses Rust 1.90.0 for potential WebAssembly compilation and performance-critical components.

### Rust Version
- rustc: 1.90.0 (1159e78c4 2025-09-14)
- cargo: 1.90.0 (840b83a10 2025-07-30)

### WebAssembly Targets

The following wasm32 targets are configured:
- `wasm32-wasip1-threads` - WASI Preview 1 with thread support
- `wasm32-wasip2` - WASI Preview 2
- `wasm32v1-none` - Bare WebAssembly without WASI

### Installing Rust and Targets

If you need to install Rust and the WebAssembly targets:

```bash
# Install rustup (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# The rust-toolchain.toml file will automatically configure the correct version
# and targets when you enter the project directory

# Or manually install targets:
rustup target add wasm32-wasip1-threads
rustup target add wasm32-wasip2
rustup target add wasm32v1-none
```

### Verifying Installation

```bash
# Check Rust version
rustc --version
# Should output: rustc 1.90.0 (1159e78c4 2025-09-14)

cargo --version
# Should output: cargo 1.90.0 (840b83a10 2025-07-30)

# Check available wasm32 targets
rustup target list | grep wasm32
# Should include:
# wasm32-wasip1-threads
# wasm32-wasip2
# wasm32v1-none
```

## Quick Start

1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Verify Rust installation:
   ```bash
   rustc --version
   cargo --version
   ```

3. Run the quick start script:
   ```bash
   cd chaosbf
   ./quickstart.sh
   ```

## Troubleshooting

- If you get import errors for numpy or matplotlib, ensure you've run `pip install -r requirements.txt`
- If Rust commands fail, ensure rustup is installed and in your PATH
- The rust-toolchain.toml file will automatically download and configure the correct Rust version when you enter the project directory
