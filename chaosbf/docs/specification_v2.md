# ChaosBF v2.0 Formal Specification

This document provides a formal specification of the ChaosBF v2.0 language, interpreter, and thermodynamic model. It is intended for researchers and developers who wish to implement compatible interpreters or understand the precise semantics of the system.

## 1. State Variables

A ChaosBF v2.0 runtime maintains the following state variables:

| Variable | Type | Description |
|:---|:---|:---|
| `M` | `Array[uint8]` | Memory tape (default size: 65536) |
| `p` | `int` | Tape pointer |
| `O` | `string` | Output buffer |
| `E` | `float` | Energy (ℝ) |
| `T` | `float` | Temperature (ℝ⁺), bounded by `[T_min, T_max]` |
| `S` | `float` | Global entropy |
| `F` | `float` | Free energy, computed as `F = E - T·S` |
| `G` | `List[string]` | Genome bank |
| `elite` | `List[string]` | Elite genome set |
| `code` | `List[char]` | Current program code |
| `ip` | `int` | Instruction pointer |
| `stack` | `List[int]` | Loop stack for `[` and `]` matching |
| `lambda_window` | `int` | Window size for λ estimation |
| `branch_children` | `deque[int]` | Windowed history of child branches |
| `branch_parents` | `deque[int]` | Windowed history of parent branches |
| `slocal` | `float` | Cached local entropy |
| `rng` | `random.Random` | Seeded random number generator |
| `pid_integral` | `float` | Integral term for PID controller |
| `pid_prev_error` | `float` | Previous error for PID controller |

## 2. Instruction Semantics

State transitions are defined as `(M, p, O, E, T, S, code, ip, stack) → (M', p', O', E', T', S', code', ip', stack')`.

### Basic Brainfuck Operations

- **`>`**: `p' = (p + 1) % tape_size`, `E' = E - 1`
- **`<`**: `p' = (p - 1) % tape_size`, `E' = E - 1`
- **`+`**: `M'[p] = (M[p] + 1) & 0xFF`, `E' = E - 2`
- **`-`**: `M'[p] = (M[p] - 1) & 0xFF`, `E' = E + 1`
- **`.`**: `O' = O + chr(M[p])`, `E' = E - 1`
- **`,`**: `M'[p] = 0`, `E' = E - 1` (Input simplified to 0)

### Loop Operations

Let `leak = min(leak_max, 1 + len(stack)/3)`.

- **`[`**: `E' = E - leak`. If `M[p] == 0`, `ip'` jumps to matching `]`. Else, `stack'.push(ip)`.
- **`]`**: `E' = E - leak`. If `M[p] != 0`, `ip'` jumps to matching `[`. Else, `stack'.pop()`.

### Thermodynamic Operations

- **`^`**: `T' = min(T_max, T + tau)`, `E' = E - 1`
- **`v`**: `T' = max(T_min, T - tau)`, `E' = E - 1`
- **`:`**: `slocal' = H(M[p-w:p+w])`, `E' = E`
- **`;`**: `S' = S + slocal`, `E' = E` (**Correctness Fix:** No energy cost)

### Evolutionary Operations

- **`?`**: With probability `P(mut)`, `code'` is a mutated version of `code`. `E' = E - 2`. `P(mut) = min(0.95, 0.2 + 0.6*T)`.
- **`*`**: If `F > theta`, `G'.append(code)`, `code'` is mutated, `E' = E - 10`.
- **`@`**: `code'` is a crossover of `code` and a random genome from `G`. `E' = E - 6`.
- **`=`**: `code'` is an optimized version of `code`. `E'` may increase based on `ΔF` (see section 3.3).
- **`!`**: If `score > 1.0`, `elite'.append(code)`. `E' = E - 1`.

### Criticality and Observability

- **`{`**: `E' = E - (2 + leak)`. With probability `P(branch)`, `M'[p] ^= 1`. Branching decision is recorded for `λ` estimation.
- **`}`**: `E' = E - leak`.
- **`#`**: Prints state. No state change.
- **`%`**: Prints `O`. `O' = ""`, `E' = E - 1`.
- **`~`**: If `F < 0` and `elite` is not empty, `code'` is a random genome from `elite`. `E' = E + 5`.

## 3. Core Model Upgrades

### 3.1. Branching Factor (λ) Estimation

The branching factor is estimated over a sliding window to provide a more stable signal for the PID controller:

```
λ_hat = sum(branch_children) / (sum(branch_parents) + ε)
```

- `branch_children`: Deque of recent child branch counts (0 or 1).
- `branch_parents`: Deque of recent parent branch counts (always 1 for `{` op).

### 3.2. PID Controller for Criticality

Temperature is updated every `pid_interval` steps to maintain `λ ≈ 1`:

```
error = λ_hat - 1.0
integral += error
derivative = error - prev_error
dT = (kp * error) + (ki * integral) + (kd * derivative)
T' = clamp(T + dT, T_min, T_max)
```

### 3.3. Learning Operator (`=`) and Free Energy

The `learn` operator (`=`) credits energy based on the free energy gain from optimization:

1.  Estimate energy cost of code segment before (`E_before`) and after (`E_after`) optimization.
2.  `dE = E_before - E_after`.
3.  Estimate entropy change `dS` (e.g., based on change in code complexity `K(code)`).
4.  `dF = dE - T * dS`.
5.  If `dF > 0`, `E' = E + (learning_cap * dF)`.

This ensures that energy is not created from nothing, respecting thermodynamic principles.

### 3.4. Grammar-Aware Mutation

The mutation operator (`?`) preserves bracket balance with high probability:

1.  With probability `1 - wild_mutation_rate`, mutation is **safe**:
    - If the selected instruction is a bracket (`[`, `]`, `{`, `}`), it is not mutated.
    - If it is a non-bracket operator, it is replaced with another non-bracket operator.
2.  With probability `wild_mutation_rate`, mutation is **wild**, and any instruction can be replaced by any other instruction.

## 4. Reproducibility and Logging

### 4.1. Run Manifest

Each run generates a JSON manifest containing:
- `run_id`, `version`, `timestamp`
- `genome`, `genome_hash`
- All initial parameters (`E`, `T`, `seed`, etc.)
- System information

### 4.2. Trace Logging

Execution is logged to a JSONL file, with each line being a JSON object representing a state snapshot. The trace schema includes PID controller terms and temporal derivatives for advanced analysis:

- `step`, `E`, `T`, `S`, `F`, `lambda`
- `K_output`, `dS_dt`, `dK_dt`
- `pid_p`, `pid_i`, `pid_d`
- `mutations`, `replications`, etc.

### 4.3. Checkpointing

Full snapshots of the interpreter state (including tape, code, and all state variables) can be saved periodically for detailed analysis and run resumption.

## 5. MAP-Elites for Quality-Diversity

An offline MAP-Elites algorithm is provided to evolve a diverse population of elites.

### 5.1. Behavioral Dimensions

The grid is defined by three behavioral dimensions:
1.  **λ-deviation:** `|λ_hat - 1.0|` (distance from criticality)
2.  **Info-per-energy:** `K(O) / |ΔE|` (information efficiency)
3.  **Entropy slope:** `dS/dt` (rate of entropy change)

### 5.2. Algorithm

1.  Initialize grid with seed genomes.
2.  Loop:
    a. Select a random elite from the grid.
    b. Create an offspring via mutation or crossover.
    c. Evaluate the offspring's fitness and behavior.
    d. If the offspring's cell is empty or its fitness is higher than the incumbent, add it to the grid.

This process discovers a wide range of viable, high-performing programs across the behavioral space.

