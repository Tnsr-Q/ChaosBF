#!/bin/bash
# Verification script for ChaosBF environment setup

echo "=========================================="
echo "ChaosBF Environment Verification"
echo "=========================================="
echo ""

# Check Python
echo "Checking Python..."
python3 --version
if [ $? -ne 0 ]; then
    echo "❌ Python 3 not found"
    exit 1
fi
echo "✓ Python 3 found"
echo ""

# Check Python dependencies
echo "Checking Python dependencies..."
python3 -c "import numpy; print(f'  numpy {numpy.__version__}')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ numpy not installed. Run: pip install -r requirements.txt"
    exit 1
fi

python3 -c "import matplotlib; print(f'  matplotlib {matplotlib.__version__}')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ matplotlib not installed. Run: pip install -r requirements.txt"
    exit 1
fi
echo "✓ Python dependencies installed"
echo ""

# Check Rust
echo "Checking Rust..."
RUSTC_VERSION=$(rustc --version 2>/dev/null)
if [ $? -ne 0 ]; then
    echo "❌ Rust not found. Install from: https://rustup.rs"
    exit 1
fi
echo "  $RUSTC_VERSION"

CARGO_VERSION=$(cargo --version 2>/dev/null)
if [ $? -ne 0 ]; then
    echo "❌ Cargo not found"
    exit 1
fi
echo "  $CARGO_VERSION"

# Check for expected Rust version
if [[ "$RUSTC_VERSION" == *"1.90.0"* ]]; then
    echo "✓ Correct Rust version (1.90.0)"
else
    echo "⚠ Warning: Expected Rust 1.90.0, but found: $RUSTC_VERSION"
fi
echo ""

# Check wasm32 targets
echo "Checking wasm32 targets..."
TARGETS_FOUND=0

if rustup target list | grep -q "wasm32-wasip1-threads (installed)"; then
    echo "  ✓ wasm32-wasip1-threads"
    TARGETS_FOUND=$((TARGETS_FOUND + 1))
else
    echo "  ❌ wasm32-wasip1-threads not installed"
fi

if rustup target list | grep -q "wasm32-wasip2 (installed)"; then
    echo "  ✓ wasm32-wasip2"
    TARGETS_FOUND=$((TARGETS_FOUND + 1))
else
    echo "  ❌ wasm32-wasip2 not installed"
fi

if rustup target list | grep -q "wasm32v1-none (installed)"; then
    echo "  ✓ wasm32v1-none"
    TARGETS_FOUND=$((TARGETS_FOUND + 1))
else
    echo "  ❌ wasm32v1-none not installed"
fi

if [ $TARGETS_FOUND -eq 3 ]; then
    echo "✓ All required wasm32 targets installed"
else
    echo "⚠ Warning: Only $TARGETS_FOUND/3 wasm32 targets installed"
    echo "  The rust-toolchain.toml should auto-install them when you enter the directory"
fi
echo ""

# Test ChaosBF
echo "Testing ChaosBF interpreter..."
cd chaosbf 2>/dev/null || cd .
TEST_OUTPUT=$(python3 src/chaosbf.py "++." --energy 10 --steps 10 2>&1)
if [ $? -eq 0 ]; then
    echo "✓ ChaosBF interpreter working"
else
    echo "❌ ChaosBF interpreter test failed"
    echo "$TEST_OUTPUT"
    exit 1
fi
echo ""

echo "=========================================="
echo "✓ Environment verification complete!"
echo "=========================================="
echo ""
echo "All dependencies are correctly installed."
echo "You can now run ChaosBF programs!"
