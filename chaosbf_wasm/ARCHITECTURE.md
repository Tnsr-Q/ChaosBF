# ChaosBF v4.0.1 Architecture

## System Overview

ChaosBF is a thermodynamic, evolvable Brainfuck implementation with MCMC-driven quality-diversity evolution. The system spans three layers:

```
┌─────────────────────────────────────────────────────────────┐
│                     UI Layer (Browser)                       │
│  www/index.html + renderer.js + worker.js                   │
│  - Canvas rendering (tape + HUD)                            │
│  - Web Worker for background simulation                     │
│  - User controls (PID, temperature, metropolis)             │
└─────────────────────────────────────────────────────────────┘
                              ↓ WASM FFI
┌─────────────────────────────────────────────────────────────┐
│                  WASM Core (Rust)                           │
│  chaosbf_wasm/src/ - Compiled to wasm32-unknown-unknown    │
│  - state.rs: Simulation engine (22 operators)              │
│  - wasm_api.rs: Frozen ABI (60+ exports)                   │
│  - aurora.rs: Behavioral descriptor learning               │
│  - critic.rs: N-gram phenotype prediction                  │
│  - island.rs: Multi-population ecology                     │
│  - lyapunov.rs: Edge-of-chaos detection                    │
│  - repro.rs: Reproducibility spine                         │
└─────────────────────────────────────────────────────────────┘
                              ↑ Research API
┌─────────────────────────────────────────────────────────────┐
│              Python Research Layer                          │
│  chaosbf/src/ - Prototyping & experimentation              │
│  - chaosbf_v3.py: Reference implementation                 │
│  - experiments.py: Scientific workflows                    │
│  - visualize.py: Analysis & plotting                       │
└─────────────────────────────────────────────────────────────┘
```

## Data Flow

### Initialization Flow

```
User clicks "Start"
    → renderer.js: init_sim(seed, width, height, code, e0, t0)
    → WASM: SIM thread_local initialized with SimState
    → renderer.js: self_check() validates setup
    → If self_check() == 1, start animation loop
    → If self_check() == 0, display error and halt
```

### Simulation Loop

```
requestAnimationFrame callback:
    → worker.js: postMessage({type: 'step', ticks: 100})
    → Worker thread: step_sim(100)
        → For each tick:
            - Execute op at IP
            - Update thermodynamic state (E, T, S, F)
            - PID control if enabled
            - Variance shaping if enabled
            - Metropolis acceptance if enabled
        → Auto-snapshot if repro_spine enabled
    → worker.js: postMessage({type: 'update', metrics})
    → renderer.js: update HUD and canvas
```

### Metrics Update

```
Every frame:
    → metrics_ptr = get_metrics_ptr()
    → metrics_array = new Float32Array(wasm.memory.buffer, metrics_ptr, 20)
    → Read 20 floats: [steps, e, t, s, f, lambda, ...]
    → Update HUD text elements
    → No allocation on JS side (pointer stable)
```

### Tape Rendering

```
Every frame or frame % 3 == 0:
    → mem_ptr = get_mem_ptr()
    → mem_len = get_mem_len()
    → tape = new Uint8Array(wasm.memory.buffer, mem_ptr, mem_len)
    → Draw heatmap on canvas (tape[i] → RGB)
    → Highlight current pointer position
```

## Module Responsibilities

### state.rs (Core Engine)

**Purpose**: Execute ChaosBF v4.0 with 22 operators and thermodynamic bookkeeping.

**Key Types**:
- `SimState`: Main simulation state (65KB tape, 4KB code, branching history, genome bank)
- `Op`: Enum of 22 operators (Gt, Lt, Plus, Minus, ..., Percent, Tilde)

**Operators**:
- Classic BF: `><+-[].,`
- Thermodynamic: `^v:;` (heat, cool, sense entropy, emit entropy)
- Evolutionary: `?*@=!~` (mutate, replicate, crossover, learn, save elite, load elite)
- Control flow: `{}#%` (conditional branch, complexity op, info/energy ratio)

**State Updates**:
```
Each step:
    1. Fetch op_byte = code[ip]
    2. op = op_from_byte(op_byte)
    3. Sense entropy for ':' operator
    4. delta_e = delta_e(op, stack_ptr, slocal)
    5. e += delta_e
    6. execute_op(op)
    7. ip += 1, steps += 1
    8. f = e - t*s
    9. Update lambda_hat (branching factor) every 10 steps
   10. PID update every pid_interval steps
   11. Variance shaping every pid_interval steps
   12. Derivative tracking every 100 steps
```

### wasm_api.rs (ABI Layer)

**Purpose**: Frozen interface between Rust and JavaScript.

**Pattern**:
```rust
thread_local! {
    static SIM: RefCell<Option<SimState>> = RefCell::new(None);
    static AURORA: RefCell<Option<AURORADescriptors>> = RefCell::new(None);
    // ... other subsystems ...
}

fn with_sim_mut<F, R>(f: F) -> R { ... }
fn with_sim<F, R>(f: F) -> R { ... }

#[no_mangle]
pub extern "C" fn step_sim(ticks: u32) {
    with_sim_mut(|sim| { ... });
}
```

**Return Buffers**:
```rust
static mut METRICS: [f32; 20] = [0.0; 20];  // Stable pointer
static mut DESC: [f32; 2] = [0.0; 2];       // AURORA descriptors
static mut EDGE_STATS: [f32; 5] = ...       // Edge-band stats
...
```

### aurora.rs (Behavioral Descriptors)

**Purpose**: Learn compressed behavioral representations via autoencoder.

**Architecture**:
```
State trace (e.g., 50 time steps × 4 features = 200 dims)
    → Encoder: dense(200 → 32, relu) → dense(32 → 2, linear)
    → Latent: 2D descriptor space
    → Decoder: dense(2 → 32, relu) → dense(32 → 200, linear)
```

**Loss**: InfoNCE contrastive loss for temporal stability:
```
anchor = z(t)
positive = z(t+1)
negatives = [z(t+2), z(t+3), ...]
loss = -log(exp(sim(anchor, positive) / tau) / sum(exp(sim(anchor, neg) / tau)))
```

### critic.rs (Phenotype Prediction)

**Purpose**: Self-bootstrapping semantics via n-gram learning.

**Mechanism**:
```
Output: [0x41, 0x42, 0x43, ...] (bytes)
    → Extract n-grams: [(0x41, 0x42), (0x42, 0x43), ...]
    → Learn transition probabilities: P(next | context)
    → Surprise = -log(P(actual | context))
    → Fitness bonus = surprise_weight * surprise
```

**Use Case**: Reward programs that produce unexpected/novel outputs.

### island.rs (Multi-Population)

**Purpose**: Prevent premature convergence via spatial isolation.

**Model**:
```
4-8 islands, each with:
    - population: Vec<Individual>
    - fitness cache
    - local novelty archive

Every migration_interval steps:
    - Compute novelty deficit: (avg_population_novelty - island_novelty)
    - If deficit < 0: emit best individual
    - If deficit > 0: accept immigrant from another island
```

### lyapunov.rs (Edge Detection)

**Purpose**: Tag individuals as critical-stable, critical-chaotic, or marginal (edge).

**Method**:
```
1. Run program with state S0
2. Run perturbed twin with S0 + ε
3. Measure divergence d(t) = ||S(t) - S'(t)||
4. Lyapunov exponent λ ≈ log(d(T)) / T
5. Bootstrap 100 runs → confidence interval [λ_low, λ_high]
6. Tag:
    - If λ_high < 0: critical-stable
    - If λ_low > 0: critical-chaotic
    - If 0 ∈ [λ_low, λ_high]: marginal (TRUE EDGE!)
```

### edge_band.rs (Routing)

**Purpose**: Direct evolutionary search to edge-of-chaos.

**Strategies**:
- `MarginalOnly`: Only mate marginal individuals (fallback to uniform if none)
- `MarginalWeighted`: Weight marginal elites higher

**Statistics**: Track counts of each edge tag for monitoring.

### repro.rs (Reproducibility)

**Purpose**: Publication-hard reproducibility (PNAS/Nature standard).

**Features**:
```
Snapshot:
    - step, e, t, s, f, lambda_hat
    - full code (up to 4KB)
    - tape prefix (first 1KB)
    - ptr, output_len, bank_size

Manifest:
    - run_id, timestamp, code_hash
    - seed, initial params
    - software version
    - SHA-256 of full state

Rewind:
    - Find snapshot <= target_step
    - Restore state from snapshot
    - Step forward to exact target_step
```

### rng.rs (PCG)

**Purpose**: Fast, deterministic, high-quality RNG (zero dependencies).

**Algorithm**: Permuted Congruential Generator (PCG-XSH-RR)
```rust
state = state * MUL + inc
xorshift = ((state >> 18) ^ state) >> 27
rot = state >> 59
output = (xorshift >> rot) | (xorshift << ((-rot) & 31))
```

## Python ↔ WASM Relationship

### Research → Production Pipeline

1. **Prototype in Python** (`chaosbf_v3.py`):
   - Rapid iteration on new features
   - NumPy/SciPy for numerics
   - Matplotlib for visualization
   - Jupyter notebooks for exploration

2. **Port to Rust** (`src/*.rs`):
   - Match Python semantics exactly
   - Replace NumPy with manual loops
   - Replace Serde with manual JSON
   - Optimize hot loops

3. **Validate Parity**:
   - Same seed → identical outputs
   - Compare metrics arrays
   - Regression tests

4. **Deploy to WASM**:
   - Compile to wasm32-unknown-unknown
   - Integrate with browser UI
   - Profile performance

### Feature Parity Matrix

| Feature | Python | Rust/WASM | Notes |
|---------|--------|-----------|-------|
| ChaosBF v4.0 (22 ops) | ✅ | ✅ | Exact match |
| PID control | ✅ | ✅ | Dual-loop (PID + variance) |
| Metropolis MCMC | ✅ | ✅ | Acceptance tracking |
| AURORA autoencoder | ✅ | ✅ | Manual backprop |
| Lyapunov estimation | ✅ | ✅ | Bootstrap CI |
| Edge-band routing | ✅ | ✅ | Marginal selection |
| Island ecology | ✅ | ✅ | Novelty-deficit migration |
| Critic n-gram | ✅ | ✅ | Surprise bonuses |
| Repro spine | ✅ | ✅ | Snapshots + manifests |
| Landauer costing | ✅ | ✅ | Entropy window |

## UI Components

### index.html

**Purpose**: Main visualization page.

**Structure**:
```html
<canvas id="tape-canvas">  <!-- Heatmap of tape memory -->
<div id="hud">             <!-- Metrics display -->
<div id="controls">        <!-- Sliders for PID, T, etc. -->
<button id="start">        <!-- Init + run simulation -->
<button id="snapshot">     <!-- Manual snapshot -->
<button id="rewind">       <!-- Time travel -->
```

### renderer.js

**Responsibilities**:
- Load WASM module
- Initialize simulation on button click
- Call `self_check()` and halt if failed
- requestAnimationFrame loop
- Read metrics via `get_metrics_ptr()`
- Draw tape heatmap
- Update HUD text
- Handle user controls (send config to WASM)

**Pointer Caching**:
```javascript
// Cache pointers (stable across calls)
const metricsPtr = wasm.exports.get_metrics_ptr();
const memPtr = wasm.exports.get_mem_ptr();

// Every frame: wrap in typed arrays (no allocation)
const metrics = new Float32Array(wasm.memory.buffer, metricsPtr, 20);
const tape = new Uint8Array(wasm.memory.buffer, memPtr, wasm.exports.get_mem_len());
```

### worker.js

**Purpose**: Offload `step_sim` to background thread (avoid blocking UI).

**Pattern**:
```javascript
// Main thread → Worker
postMessage({type: 'step', ticks: 100});

// Worker: run simulation
onmessage = (e) => {
    wasm.exports.step_sim(e.data.ticks);
    const metrics = readMetrics();
    postMessage({type: 'update', metrics});
};
```

## Performance Characteristics

### Hot Path

```
step_sim(1000) ≈ 1ms @ 3GHz
    → 1000 iterations of step()
    → Each step: ~3000 cycles
        - Array access (code, mem): ~100 cycles
        - Arithmetic (e, t, s): ~50 cycles
        - Branching (match op): ~20 cycles
        - Stack ops: ~30 cycles
```

### Cold Path

```
aurora_compute_descriptors() ≈ 5ms
    → Encode 50-step trace: 200 → 2 dims
    → Matrix mul: O(200 * 32 + 32 * 2) ≈ 6500 FLOPs

critic_compute_fitness() ≈ 1ms
    → Scan output buffer: O(output_len)
    → HashMap lookups: O(log n) per n-gram
```

### Memory Footprint

```
SimState: ~180 KB
    - mem: 65536 bytes
    - code: 4096 bytes
    - genome_bank: 100 × 4096 = 400KB (shared)
    - elite: 50 × 4096 = 200KB
    - output_buffer: 16384 bytes

Total WASM binary: ~275 KB (optimized release)
```

## Determinism & Reproducibility

### Guarantees

1. **Seed Determinism**: Same seed → identical simulation
2. **Pointer Stability**: Metrics/mem pointers never change (within instance)
3. **Snapshot Rewind**: Restore state to any snapshotted step

### Non-Determinism Sources (Eliminated)

- ❌ OS entropy (e.g., `/dev/urandom`)
- ❌ Timestamp-based seeds
- ❌ Floating-point non-associativity (use consistent order)
- ❌ Hash map iteration order (use BTreeMap or sorted keys)

### Validation Test

```bash
# Run twice with same seed
wasm.exports.init_sim(12345, 256, 256, code_ptr, code_len, 200.0, 0.6);
wasm.exports.step_sim(1000);
metrics1 = readMetrics();

wasm.exports.init_sim(12345, 256, 256, code_ptr, code_len, 200.0, 0.6);
wasm.exports.step_sim(1000);
metrics2 = readMetrics();

assert(metrics1 == metrics2);  // Bit-exact equality
```

## Future Extensions (v5.0+)

### GPU Acceleration

- `www/gpu/ae_encode.wgsl`: AURORA encoder in WGSL
- `www/gpu/critic_ngram.wgsl`: Parallel n-gram counting
- `www/gpu/engine.js`: WebGPU compute pipeline

### Distributed Search

- `www/mesh.js`: WebSocket protocol for multi-client evolution
- `server/mesh-server.js`: Node.js coordinator (island migration)

### Telemetry

- `www/telemetry.js`: WebRTC DataChannel for real-time metrics streaming

### Gallery

- `www/gallery.html`: Browse snapshots from IndexedDB
- `www/gallery.js`: Load and replay historical runs

---

**Version**: 4.0.1
**Last Updated**: 2025-10-29
**Maintainer**: ChaosBF Team
