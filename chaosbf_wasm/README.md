# ChaosBF WASM - Real-time Evolution Visualization

A high-performance WASM port of the ChaosBF thermodynamic programming language with real-time visualization like Conway's Life.

## Features

- **Rust/WASM Core**: Blazing-fast interpreter kernel compiled to WebAssembly
- **Web Worker**: Non-blocking simulation running in background thread
- **Canvas 2D Renderer**: Smooth 60 FPS visualization with ImageData streaming
- **Thermodynamic Model**: Energy (E), Temperature (T), Entropy (S), Free Energy (F)
- **Evolution Operators**: Mutation, Replication, Crossover, Learning, Elite Selection
- **Criticality Control**: Self-organizing branching factor λ ≈ 1
- **Landauer Costing**: Information erasure thermodynamics

## Architecture

```
chaosbf_wasm/
├─ Cargo.toml          # Rust WASM crate config
├─ src/
│  ├─ lib.rs           # WASM bindings & main interpreter
│  ├─ state.rs         # Simulation state (tape, E/T/S/F, λ, PRNG)
│  ├─ ops.rs           # Opcode table, ΔE costs, branching depth leak
│  └─ thermo.rs        # Landauer window entropy, Metropolis gate
└─ www/
   ├─ index.html       # Main UI with controls
   ├─ worker.js        # Web Worker (background simulation)
   ├─ renderer.js      # Canvas 2D renderer
   └─ pkg/             # Built WASM output (auto-generated)
```

## Quick Start

### Prerequisites

- Rust (1.70+)
- wasm-pack: `cargo install wasm-pack`
- Python 3 (for local dev server)

### Build & Run

```bash
# Build WASM module
wasm-pack build --target web --out-dir www/pkg

# Serve locally
python3 -m http.server 8080 -d www

# Open browser to http://localhost:8080
```

Or use npm scripts:

```bash
npm install
npm run build    # Build WASM
npm run serve    # Serve on :8080
npm run dev      # Build + serve
```

## ChaosBF Opcodes

### Core BF
- `><` : Move tape pointer left/right
- `+-` : Increment/decrement cell
- `[]` : Loop while cell != 0
- `.,` : Output/input (simplified)

### Thermodynamics
- `^v` : Anneal temperature up/down
- `:;` : Sense local entropy / Accumulate to global S

### Evolution
- `?` : Thermal mutation (p ∝ T)
- `*` : Replicate if F > θ
- `@` : Crossover with genome bank
- `=` : Learn (peephole optimization)
- `!` : Elite selection

### Control
- `{}` : Critical loop (branching factor λ control)
- `#` : Measure (display metrics)
- `%` : Phenotype dump
- `~` : Quench (revert to elite if F < 0)

## Example Programs

### Primordial Soup
```
?*{+>-<}@=
```
Random mutation + replication + critical loops + crossover + learning

### Replicator
```
[>+<-]*?@
```
Copy pattern, replicate, mutate, crossover

### Critical Dance
```
{^v:;}?*@=
```
Temperature oscillation + entropy sensing + evolution

### Learning Soup
```
=?*>+<-@
```
Optimization-driven with movement and evolution

## Performance Tuning

- **Grid Size**: 32-512 (128 recommended)
- **Ticks/Frame**: 10-1000 (100 recommended)
- **Energy**: Higher = longer runs
- **Temperature**: Higher = more exploration, lower = more exploitation

## Implementation Notes

### Zero-Copy Frame Transfer
Frames are transferred between Worker and main thread using Transferable ArrayBuffers - no copying overhead.

### Landauer Entropy Costing
When a cell write reduces local Shannon entropy (information erasure), energy is charged proportional to ΔH × T (Landauer's principle).

### Deterministic PRNG
Uses ChaCha20 for reproducible runs given the same seed.

## License

MIT
