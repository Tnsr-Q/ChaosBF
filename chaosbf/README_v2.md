# ChaosBF v2.0: A Production-Grade Thermodynamic, Evolvable Language

![MAP-Elites Grid](/home/ubuntu/chaosbf/output/map_elites_visualization.png)

**ChaosBF (CBF) v2.0** is a research-grade implementation of a thermodynamic, evolvable Brainfuck derivative. It is designed to operate at the "edge of chaos," a regime where computation can exhibit both order and unpredictability, fostering open-ended learning and emergent behavior. This version incorporates significant upgrades for correctness, robustness, and advanced research capabilities, elevating it from a conceptual toy to a state-of-the-art laboratory for studying artificial life and complex adaptive systems.

## v2.0: Key Upgrades and Features

This version represents a major overhaul, focusing on scientific rigor and advanced evolutionary dynamics:

- **Thermodynamic Correctness:** The energy model has been fixed to prevent double-penalizing free energy. The entropy accumulation operator (`;`) is now a pure entropy operation (ΔE=0), and the learning operator (`=`) credits energy based on a thermodynamically sound free energy gain (ΔF).

- **PID-Controlled Criticality:** A closed-loop PID controller has been implemented to automatically adjust system temperature (`T`) to maintain the branching factor (λ) near the critical value of 1.0. This ensures sustained operation at the edge of chaos.

- **MAP-Elites for Quality-Diversity:** The system now includes a MAP-Elites algorithm (`src/map_elites.py`) to evolve a diverse population of high-performing programs across multiple behavioral dimensions (criticality, information efficiency, entropy slope).

- **Grammar-Aware Mutation:** The mutation operator (`?`) is now grammar-aware, preserving bracket balance by default to create more viable offspring, while still allowing for a configurable rate of "wild" mutations to explore chaotic regions of the search space.

- **Reproducibility and Logging:** The new CLI (`src/cli.py`) provides deterministic execution via RNG seeding, generates a JSON manifest for each run, and logs detailed traces in JSONL format, making experiments citable and reproducible.

- **Systematic Experiment Protocols:** A dedicated experiment runner (`src/experiments.py`) allows for systematic analysis, including criticality sweeps, energy-budget ablations, and operator-knockout studies.

- **Invariant-Based Testing:** A new test suite (`tests/test_invariants.py`) uses property-based testing to validate thermodynamic invariants and system properties, ensuring the implementation adheres to its formal specification.

## Project Structure

```
/home/ubuntu/chaosbf/
├── src/
│   ├── chaosbf_v2.py        # Core v2.0 interpreter
│   ├── map_elites.py        # MAP-Elites implementation
│   ├── experiments.py       # Experiment protocols
│   └── cli.py               # Enhanced CLI with logging
├── tests/
│   └── test_invariants.py   # Property-based invariant tests
├── docs/
│   ├── specification_v2.md  # Formal specification for v2.0
│   └── research_protocols.md# Research protocols and experiment guide
├── output/
│   └── (manifests, logs, plots, checkpoints)
└── README_v2.md
```

## Getting Started

The enhanced CLI provides a unified interface for running programs and experiments.

### Running a Program

Use the `run` command to execute a ChaosBF program with full logging and reproducibility features.

```bash
# From the project root directory
cd /home/ubuntu/chaosbf

# Run the hot seed with a specific seed for reproducibility
python3 src/cli.py run "++[>+<-].:{;}{?}^*=@=.#%" --seed 42 --steps 20000 --export output/my_run
```

This command will generate:
- `output/my_run/manifest_{run_id}.json`: Metadata for the run.
- `output/my_run/trace_{run_id}.jsonl`: Detailed step-by-step log.
- `output/my_run/final_{run_id}.json`: Final state snapshot.

### Running Experiments

The `experiments.py` script provides a way to run systematic analyses.

```bash
# Run a criticality sweep on the hot seed genome
python3 src/experiments.py --experiments criticality --genome "++[>+<-].:{;}{?}^*=@=.#%"

# Run all experiments
python3 src/experiments.py --experiments all
```

### Evolving with MAP-Elites

The `map_elites.py` script evolves a diverse population of programs.

```bash
# Run MAP-Elites for 1000 iterations
python3 src/map_elites.py --iterations 1000 --seed 42 --output output/map_elites_run
```

This will produce a grid of high-performing, diverse elites and a visualization of the behavior space.

## Instruction Set v2.0

The instruction set remains the same, but the underlying implementation of several operators has been improved for correctness and performance.

| Op  | Meaning                                          | ΔE Cost        | v2.0 Notes                                     |
|:----|:-------------------------------------------------|:---------------|:-----------------------------------------------|
| `>` | `ptr++`                                          | -1             |                                                |
| `<` | `ptr--`                                          | -1             |                                                |
| `+` | `M[p]++`                                         | -2             |                                                |
| `-` | `M[p]--`                                         | +1             |                                                |
| `[` | `while M[p] != 0`                                | -(1+d/3)       | Loop leak now capped by `leak_max`             |
| `]` | `end while`                                      | -(1+d/3)       |                                                |
| `.` | `emit M[p] to O`                                 | -1             |                                                |
| `,` | `read byte -> M[p]`                              | -1             |                                                |
| `^` | `T += τ`                                         | -1             | Manual temperature control (overridden by PID) |
| `v` | `T -= τ`                                         | -1             | Manual temperature control (overridden by PID) |
| `:` | Sense local entropy `H(M[p-w:p+w])`               | 0              |                                                |
| `;` | Accrue local entropy to global `S`               | **0**          | **FIXED:** Pure entropy op, no energy cost       |
| `?` | Thermal mutate one random instruction            | -2             | **UPGRADED:** Grammar-aware by default         |
| `*` | Replicate current genome to `G`                  | -10            |                                                |
| `@` | Crossover with random gene from `G`              | -6             |                                                |
| `=` | Learn: peephole optimization                     | `+ΔF_gain`     | **UPGRADED:** Energy gain based on `ΔF`          |
| `!` | Select: promote to elite set if score is high    | -1             |                                                |
| `{` | Start critical loop                              | -2 / iter      | Now feeds a more stable windowed `λ` estimator |
| `}` | End critical loop                                | 0              |                                                |
| `#` | Measure: print `E,T,S,F,λ`                       | 0              |                                                |
| `%` | Phenotype dump: print `O` then clear             | -1             |                                                |
| `~` | Quench: revert to elite ancestor if `F < 0`      | +5             |                                                |

## Conclusion

ChaosBF v2.0 is a significant step towards a robust and reliable platform for research in artificial life, open-ended evolution, and complex systems. The focus on thermodynamic correctness, closed-loop control, and reproducible science makes it a powerful tool for exploring the fundamental principles of computation and life.

