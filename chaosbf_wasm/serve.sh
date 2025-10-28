#!/bin/bash
# Quick launch script for ChaosBF WASM visualization

echo "🦀 ChaosBF WASM - Starting local server..."
echo ""
echo "Open your browser to:"
echo "  http://localhost:8080/index_standalone.html"
echo ""
echo "Press Ctrl+C to stop"
echo ""

cd "$(dirname "$0")/www"
python3 -m http.server 8080
