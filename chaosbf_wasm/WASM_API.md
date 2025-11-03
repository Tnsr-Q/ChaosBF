# ChaosBF v4.0.1 WASM API Documentation

## Overview

This document defines the frozen Application Binary Interface (ABI) for ChaosBF WASM module. All exports maintain backward compatibility and pointer stability guarantees.

## Memory Model

### Pointer Stability

All return pointers remain constant across calls:
- `get_mem_ptr()` → stable tape pointer
- `get_metrics_ptr()` → stable 20×f32 array
- `get_output_ptr()` → stable output buffer
- `aurora_compute_descriptors()` → stable 2×f32 array
- `edge_band_get_stats_ptr()` → stable 5×f32 array
- `ecology_get_stats_ptr()` → stable 4×f32 array
- `critic_compute_fitness()` → stable 2×f32 array

**Important**: Pointers are stable within a single WASM instance. Do not cache pointers across module reloads.

### Memory Layout

```
Tape Memory:     65536 bytes max (actual size returned by get_mem_len())
Code Buffer:     4096 bytes max
Output Buffer:   16384 bytes max
```

## Core Simulation API

### Initialization

```c
void init_sim(
    uint64_t seed,
    size_t width,
    size_t height,
    const uint8_t* code_ptr,
    size_t code_len,
    float e0,
    float t0
)
```

Initializes the simulation state. Must be called before any other function.

**Parameters:**
- `seed`: RNG seed for deterministic runs
- `width`, `height`: Tape dimensions (product ≤ 65536)
- `code_ptr`: Initial Brainfuck program
- `code_len`: Program length (max 4096)
- `e0`: Initial energy
- `t0`: Initial temperature

### Execution

```c
void step_sim(uint32_t ticks)
```

Executes `ticks` simulation steps. Stops early if energy reaches 0 or code is empty.

### Memory Access

```c
const uint8_t* get_mem_ptr()
size_t get_mem_len()
```

Returns pointer to tape memory and its size in bytes.

```c
const uint8_t* get_output_ptr()
size_t get_output_len()
```

Returns pointer to output buffer and current output length.

### Metrics

```c
const float* get_metrics_ptr()
```

Returns pointer to 20-element float array:

| Index | Metric | Description |
|-------|--------|-------------|
| 0 | steps | Total simulation steps |
| 1 | e | Current energy |
| 2 | t | Current temperature |
| 3 | s | Current entropy |
| 4 | f | Free energy (F = E - TS) |
| 5 | lambda_hat | Branching factor estimate |
| 6 | mutations | Total mutations performed |
| 7 | replications | Total replications |
| 8 | crossovers | Total crossover operations |
| 9 | learns | Total learning operations |
| 10 | lambda_volatility | λ volatility (std dev) |
| 11 | ds_dt_ema | Entropy time derivative (EMA) |
| 12 | dk_dt_ema | Complexity time derivative (EMA) |
| 13 | complexity_estimate | Kolmogorov complexity proxy |
| 14 | info_per_energy | Information efficiency |
| 15 | bank_size | Genome bank size |
| 16 | output_len | Current output length |
| 17 | pid_kp | PID proportional gain |
| 18 | variance_gamma | Variance shaping parameter |
| 19 | acceptance_rate | Metropolis acceptance fraction |

## Configuration API

### PID Controller

```c
void set_pid_params(float kp, float ki, float kd, bool enable)
```

Configures dual-loop PID controller for criticality maintenance (λ → 1.0).

### Variance Shaping

```c
void set_variance_shaping(float gamma, bool enable)
```

Enables second control loop targeting λ variance.

### Metropolis MCMC

```c
void set_metropolis(bool enable)
```

Enables/disables Metropolis acceptance criterion for mutations.

## AURORA API (Learned Behavioral Descriptors)

```c
void aurora_init(size_t trace_length, size_t state_features, size_t latent_dim, uint64_t seed)
const float* aurora_compute_descriptors()
```

**Returns**: 2-element float array [descriptor1, descriptor2]

AURORA learns behavioral descriptors via autoencoder with InfoNCE contrastive loss.

## Lyapunov & Edge-Band API

```c
void lyapunov_init(float perturbation, size_t window_size)
void edge_band_init(float marginal_weight, uint64_t seed)
const float* edge_band_get_stats_ptr()
```

**Returns**: 5-element float array [critical_stable, critical_chaotic, marginal, total, marginal_fraction]

Edge-band routing preferentially selects "marginal" elites near the edge of chaos.

## Island Ecology API

```c
void ecology_init(size_t n_islands, uint64_t seed)
void ecology_evolve(uint32_t steps, uint32_t migration_interval)
const float* ecology_get_stats_ptr()
```

**Returns**: 4-element float array [generation, population, immigrants, emigrants]

Multi-island speciation with novelty-deficit migration.

## Critic-in-the-Loop API

```c
void critic_init(size_t ngram_size, float surprise_weight, size_t population_size)
const float* critic_compute_fitness()
void critic_learn_from_output()
```

**Returns**: 2-element float array [total_fitness, surprise_bonus]

N-gram phenotype prediction with surprise-based fitness bonuses.

## Reproducibility Spine API

```c
void repro_init(uint32_t snapshot_interval, bool enable_crash_capsule)
void repro_start_run(const uint8_t* run_id_ptr, size_t run_id_len)
void repro_snapshot()
size_t repro_snapshot_count()
bool repro_rewind(uint32_t target_step)
```

Publication-hard reproducibility with snapshots, manifests, and rewind capability.

**Note**: Snapshots save first 1KB of tape for memory efficiency. Full state captured every `snapshot_interval` steps.

## Validation API

```c
int32_t self_check()
```

**Returns**:
- `1` if all invariants pass
- `0` if any check fails

**Checks:**
1. Metropolis acceptance rate ∈ (0.20, 0.30) after 100+ samples
2. Pointer stability (metrics pointer unchanged)
3. State consistency (bounds checking)

Call after `init_sim` to validate setup. If returns `0`, halt simulation and report error.

## Determinism Guarantees

- Same seed → identical simulation
- All RNG seeded with user-provided seed
- No external entropy sources
- Single-threaded execution model

**Reproducibility Test**: Run with seed=12345, first 100 steps should produce identical metrics array.

## Error Handling

Functions expecting initialized state will panic with descriptive message if called before `init_sim`. Check `self_check()` return value to validate initialization.

## Performance Notes

- Hot loop: `step_sim` with 100-1000 tick batches
- Avoid calling getters every frame (cache pointers)
- Metrics update is cheap (no allocation)
- AURORA/Critic updates are O(n) in trace length

## ABI Changelog

### v4.0.1
- Replaced `static mut` with thread-local storage
- Added `get_mem_len()` for tape size queries
- Added `self_check()` validation function
- Fixed acceptance_rate metric (index 19)
- Froze all exports in `wasm_api.rs`

### v4.0.0
- Initial stable release
