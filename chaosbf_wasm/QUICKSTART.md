# ChaosBF WASM - Quick Start Guide

## ⚡ Fast Start (Already Built!)

The WASM module is already compiled and ready to run:

```bash
./serve.sh
```

Then open your browser to **http://localhost:8080/index_standalone.html**

## 🎮 Using the Interface

### Presets
Click any preset to load a pre-configured "soup":
- **Primordial Soup**: Random mutation + evolution
- **Replicator**: Self-copying patterns
- **Critical Dance**: Temperature oscillations at λ≈1
- **Learning Soup**: Optimization-driven evolution

### Controls
- **▶ Start**: Begin simulation
- **■ Stop**: Pause simulation
- **⟲ Reset**: Reset with current settings
- **Speed**: Adjust simulation speed (10-1000 ticks/frame)

### Parameters
- **Grid Size**: 32-256 (128 recommended)
- **Initial Code**: ChaosBF program to evolve
- **Energy (E₀)**: Starting energy (higher = longer runs)
- **Temperature (T₀)**: Thermal noise level (higher = more exploration)
- **Seed**: RNG seed for reproducibility

### Thermodynamic State
Watch the real-time evolution metrics:
- **E**: Energy remaining
- **T**: Temperature (annealed by `^v` ops)
- **S**: Entropy accumulator
- **F = E - T·S**: Free energy (drives replication)
- **λ**: Branching factor (criticality measure)

### Evolution Stats
- **Mutations**: Random code changes (`?`)
- **Replications**: Genome copies (`*`)
- **Crossovers**: Genetic recombination (`@`)
- **Learns**: Peephole optimizations (`=`)

## 🔬 Example Experiments

### 1. Watch Criticality Emerge
```
Code: {^v:;}?*@=
E₀: 250
T₀: 0.5
Speed: 100
```
Observe λ (branching factor) self-organize toward 1.0

### 2. Optimize a Pattern
```
Code: +>-<+>-<+>-<=
E₀: 400
T₀: 0.3
Speed: 50
```
Watch peephole optimization (`=`) simplify redundant ops

### 3. Explosive Replication
```
Code: *?@
E₀: 500
T₀: 0.8
Speed: 200
```
High temperature drives rapid mutation + replication

### 4. Stable Oscillator
```
Code: [>+<-]^v
E₀: 300
T₀: 0.2
Speed: 100
```
Low temperature creates stable cyclic behavior

## 🧬 ChaosBF Opcode Reference

### Core BF
- `><` Move tape pointer
- `+-` Inc/dec cell value
- `[]` Loop while cell != 0
- `.,` Output/input

### Thermodynamics
- `^v` Raise/lower temperature
- `:` Sense local entropy
- `;` Accumulate entropy to S

### Evolution
- `?` Mutate (probability ∝ T)
- `*` Replicate if F > θ
- `@` Crossover with genome bank
- `=` Learn (peephole optimize)
- `!` Promote to elite

### Control
- `{}` Critical branching loop
- `#` Measure (display state)
- `%` Phenotype dump
- `~` Quench (revert to elite)

## 🎨 Visualization

The canvas shows the memory tape as a 2D grid:
- **Color**: Cell value (0-255) mapped to HSV palette
- **Bright regions**: High values (active computation)
- **Dark regions**: Zeros (inactive/erased)
- **Patterns**: Emergent structures from evolution

Look for:
- **Gliders**: Moving patterns (rare!)
- **Oscillators**: Repeating structures
- **Replicators**: Self-copying regions
- **Noise**: High-entropy chaos

## 🏗️ Rebuilding (Optional)

If you modify the Rust source:

```bash
# 1. Rebuild WASM
cargo build --target wasm32-unknown-unknown --release

# 2. Copy to www
cp target/wasm32-unknown-unknown/release/chaosbf_wasm.wasm www/pkg/

# 3. Serve
./serve.sh
```

## 📊 Performance Tips

- **Grid 128×128** @ 100 ticks/frame = smooth 60 FPS
- **Grid 256×256** @ 50 ticks/frame = detailed, slower
- **Grid 64×64** @ 500 ticks/frame = fast evolution

WASM is fast! The bottleneck is usually Canvas rendering, not the simulation.

## 🐛 Troubleshooting

**Black screen?**
- Check browser console for errors
- Ensure worker_standalone.js is served (not blocked by CORS)
- Try Chromium/Chrome (best WASM support)

**Simulation stops immediately?**
- Energy depleted (E ≤ 0) - increase E₀
- Invalid code - try a preset

**Slow rendering?**
- Reduce grid size
- Lower ticks/frame
- Close other tabs

## 🚀 Advanced: Custom Programs

Try building your own thermodynamic programs:

**Entropy Harvester**
```
[:;]?*@  # Sense & accumulate entropy, mutate, replicate
```

**Temperature Controller**
```
{^v}[>+<-]  # Critical branching + stable loop
```

**Learning Engine**
```
=?*@[>+<-][<+>-]  # Learn, evolve, redundant ops to optimize
```

**Quench Cycle**
```
?*@~  # Mutate, replicate, crossover, quench (oscillate)
```

---

**Enjoy watching thermodynamic evolution unfold in real-time!** 🔥🧬
