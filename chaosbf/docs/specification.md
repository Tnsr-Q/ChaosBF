# ChaosBF Technical Specification

This document provides a formal specification of the ChaosBF language and its implementation.

## 1. State Variables

A ChaosBF runtime maintains the following state:

| Variable | Type | Description |
|:---------|:-----|:------------|
| `M` | Array[uint8] | Memory tape (default size: 65536 cells) |
| `p` | int | Tape pointer (0 to tape_size-1) |
| `O` | string | Output buffer |
| `E` | float | Energy (ℝ) |
| `T` | float | Temperature (ℝ⁺) |
| `S` | float | Entropy |
| `F` | float | Free energy, computed as `F = E - T·S` |
| `G` | List[string] | Genome bank (population of code segments) |
| `elite` | List[string] | Elite genome set |
| `code` | List[char] | Current program code |
| `ip` | int | Instruction pointer |
| `stack` | List[int] | Loop stack for `[` and `]` matching |
| `branch_hist` | List[int] | History of branching decisions for λ calculation |
| `slocal` | float | Cached local entropy value |
| `w` | int | Window size for entropy sensing (default: 8) |
| `tau` | float | Temperature adjustment step (default: 0.1) |
| `theta` | float | Free energy threshold for replication (default: 5.0) |

## 2. Instruction Set

The ChaosBF instruction set consists of 22 operators, each with specific semantics and energy costs.

### Basic Brainfuck Operations

| Op | Semantics | ΔE | Notes |
|:---|:----------|:---|:------|
| `>` | `p = (p + 1) mod tape_size` | -1 | Move pointer right |
| `<` | `p = (p - 1) mod tape_size` | -1 | Move pointer left |
| `+` | `M[p] = (M[p] + 1) mod 256` | -2 | Increment cell |
| `-` | `M[p] = (M[p] - 1) mod 256` | +1 | Decrement cell (dissipative) |
| `.` | `O += chr(M[p])` | -1 | Output byte |
| `,` | `M[p] = read_input()` | -1 | Read input (simplified to 0) |
| `[` | If `M[p] == 0`, jump to matching `]`; else push `ip` to stack | -(1 + depth/3) | Loop start with depth-dependent leak |
| `]` | If `M[p] != 0`, jump to matching `[`; else pop stack | -(1 + depth/3) | Loop end |

### Thermodynamic Operations

| Op | Semantics | ΔE | Notes |
|:---|:----------|:---|:------|
| `^` | `T += tau` | -1 | Anneal up (increase temperature) |
| `v` | `T = max(0.01, T - tau)` | -1 | Anneal down (decrease temperature) |
| `:` | `slocal = H(M[p-w:p+w])` | 0 | Sense local entropy |
| `;` | `S += slocal` | -slocal | Accrue local entropy to global |

### Evolutionary Operations

| Op | Semantics | ΔE | Notes |
|:---|:----------|:---|:------|
| `?` | Mutate one random instruction with probability `σ(T)` | -2 | Thermal mutation |
| `*` | If `F > theta`, replicate genome to `G` | -10 | Replication (conditional) |
| `@` | Crossover with random genome from `G` | -6 | Sexual reproduction |
| `=` | Peephole optimization (remove `+-`, `-+`, `><`, `<>`) | +gain*0.5 | Learning |
| `!` | If `score > 1.0`, add to elite set | -1 | Selection |

### Critical and Observability Operations

| Op | Semantics | ΔE | Notes |
|:---|:----------|:---|:------|
| `{` | Toggle `M[p]` with probability `β(T)`, record branching | -2 | Critical loop start |
| `}` | End critical loop | 0 | Critical loop end |
| `#` | Print `E, T, S, F, λ` | 0 | Measurement |
| `%` | Print `O`, then clear `O` | -1 | Phenotype dump |
| `~` | If `F < 0` and elite exists, revert to random elite genome | +5 | Quench (survival fallback) |

## 3. Energy Model

The energy model is central to ChaosBF's dynamics. Each instruction has an associated energy cost `ΔE`, which is applied to the system's energy `E` before the instruction is executed.

### Energy Cost Formula

For most instructions, the energy cost is a fixed constant. For loop operations (`[`, `]`, `{`, `}`), there is an additional depth-dependent "leak":

```
ΔE_loop = base_cost - (1 + depth/3)
```

where `depth` is the current nesting level of loops.

### Halting Condition

The program halts when:
- `E ≤ 0`, or
- The maximum number of execution steps is reached

## 4. Thermodynamic Quantities

### Entropy Calculation

Local entropy is calculated using Shannon entropy over a sliding window of the memory tape:

```
H = -Σ (p_i * log₂(p_i))
```

where `p_i` is the probability of byte value `i` in the window `M[p-w:p+w]`.

### Free Energy

Free energy is the Helmholtz free energy from thermodynamics:

```
F = E - T·S
```

This quantity represents the energy available for useful work. High free energy indicates a system with both high energy and low entropy (high order).

### Branching Factor

The branching factor `λ` is estimated as the mean of the branching history:

```
λ = mean(branch_hist)
```

where `branch_hist` records whether critical branches were taken (1) or not (0).

## 5. Evolutionary Dynamics

### Mutation

The mutation operator `?` has a temperature-dependent probability:

```
P(mutation) = min(0.95, 0.2 + 0.6 * T)
```

When a mutation occurs, a random instruction in the code is replaced with a random operator from the instruction set.

### Replication

The replication operator `*` is conditional on free energy:

```
if F > theta:
    G.append(code)
    for i in range(max(1, int(T * 3))):
        mutate_random_position()
```

Replication includes a "mutation burst" proportional to temperature, introducing variation into the offspring.

### Crossover

The crossover operator `@` performs single-point crossover:

```
if len(G) > 0:
    parent = random.choice(G)
    cut = random.randint(1, min(len(code), len(parent)) - 1)
    code = code[:cut] + parent[cut:]
```

### Learning

The learning operator `=` performs peephole optimization by removing canceling instruction pairs:

```
code = code.replace('+-', '').replace('-+', '')
code = code.replace('><', '').replace('<>', '')
E += (original_length - new_length) * 0.5
```

### Selection

The selection operator `!` computes a fitness score:

```
score = K(O) / (1 + |E - E_initial|)
```

where `K(O)` is the Kolmogorov complexity proxy (compression-based) of the output.

If `score > 1.0`, the current genome is added to the elite set.

## 6. Criticality Control

The critical loop operators `{` and `}` are designed to maintain the system near the edge of chaos.

Within a `{` block, the system randomly toggles the current cell with probability:

```
P(toggle) = min(0.9, 0.3 + 0.4 * T)
```

This creates a stochastic branching structure. The branching factor `λ` is tracked, and the system can adjust its temperature to maintain `λ ≈ 1`.

## 7. Implementation Notes

### Instruction Pointer Wrapping

The instruction pointer wraps around when it reaches the end of the code:

```
ip = ip % len(code)
```

This allows programs to loop indefinitely (until energy is depleted).

### Loop Matching

The `[` and `]` operators use a stack-based mechanism for matching. When a `[` is encountered and the current cell is non-zero, the instruction pointer is pushed onto the stack. When a `]` is encountered and the current cell is non-zero, the instruction pointer jumps back to the matching `[`.

### Complexity Proxy

The Kolmogorov complexity proxy `K(x)` is approximated using compression:

```
K(x) = max(0, len(x) - len(compress(x)))
```

where `compress()` uses the zlib compression algorithm.

## 8. Example Execution Trace

Consider the simple program `+++[>+<-].`:

| Step | ip | Op | M[0] | M[1] | E | Notes |
|:-----|:---|:---|:-----|:-----|:--|:------|
| 0 | 0 | `+` | 0 → 1 | 0 | 200 → 198 | Increment M[0] |
| 1 | 1 | `+` | 1 → 2 | 0 | 198 → 196 | Increment M[0] |
| 2 | 2 | `+` | 2 → 3 | 0 | 196 → 194 | Increment M[0] |
| 3 | 3 | `[` | 3 | 0 | 194 → 193 | M[0] != 0, enter loop |
| 4 | 4 | `>` | 3 | 0 | 193 → 192 | Move to M[1] |
| 5 | 5 | `+` | 3 | 0 → 1 | 192 → 190 | Increment M[1] |
| 6 | 6 | `<` | 3 | 1 | 190 → 189 | Move to M[0] |
| 7 | 7 | `-` | 3 → 2 | 1 | 189 → 190 | Decrement M[0] (gains energy) |
| 8 | 8 | `]` | 2 | 1 | 190 → 189 | M[0] != 0, jump to `[` |
| ... | ... | ... | ... | ... | ... | Loop continues |
| N | 8 | `]` | 0 | 3 | ... | M[0] == 0, exit loop |
| N+1 | 9 | `.` | 0 | 3 | ... | Output chr(0) |

This trace illustrates the basic execution model and the energy dynamics of a simple loop.

