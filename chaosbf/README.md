# ChaosBF: A Thermodynamic, Evolvable Brainfuck

![Thermodynamics Plot](/home/ubuntu/chaosbf/output/demo_thermodynamics.png)

**ChaosBF (CBF)** is a derivative of the esoteric programming language Brainfuck, extended with concepts from thermodynamics, evolution, and complex systems. It is designed to operate at the "edge of chaos," a regime where computation can exhibit both order and unpredictability, fostering open-ended learning and emergent behavior.

This project provides a complete implementation of the ChaosBF interpreter, along with example programs, testing tools, and visualization utilities to explore its unique dynamics.

## Key Features

- **Thermodynamic Core:** The system models `Energy (E)`, `Temperature (T)`, `Entropy (S)`, and `Free Energy (F = E - T*S)`. Every operation has an energy cost, and the program halts when energy is depleted.
- **Evolvability:** The language includes instructions for:
    - **Mutation (`?`):** Randomly alters instructions in the code, with the mutation rate influenced by temperature.
    - **Replication (`*`):** Copies the current program (genome) into a population bank, allowed only when free energy is above a certain threshold.
    - **Crossover (`@`):** Combines the current genome with another from the population, creating novel programs.
- **Learning (`=`):** A simple peephole optimization mechanism allows the program to rewrite and simplify itself, converting structural improvements into energy.
- **Criticality Control (`{}`):** A special loop structure nudges the system towards a branching factor (λ) of approximately 1, characteristic of systems at the edge of chaos.
- **Open-Ended Fitness:** The system is designed without a specific, fixed task. Its implicit goal is to maximize informational complexity (`K(output)`) per unit of energy consumed, encouraging the evolution of non-trivial behaviors.

## Project Structure

```
/home/ubuntu/chaosbf/
├── src/
│   ├── chaosbf.py         # Main interpreter implementation
│   └── visualize.py       # Visualization and tracing tools
├── examples/
│   └── seed_programs.py   # Collection of example programs
├── tests/
│   └── test_chaosbf.py      # Test suite for the interpreter
├── docs/
│   └── (documentation to be created)
├── output/
│   ├── demo_thermodynamics.png
│   ├── demo_evolution.png
│   ├── demo_criticality.png
│   └── demo_data.csv
└── README.md
```

## Getting Started

### Running the Interpreter

The main interpreter is located at `src/chaosbf.py`. You can run it from the command line:

```bash
cd /home/ubuntu/chaosbf
python3 src/chaosbf.py "<code>" [options]
```

**Example:**

```bash
python3 src/chaosbf.py "++[>+<-]." --energy 100 --steps 500
```

### Running Examples

A collection of seed programs demonstrating various features is available in `examples/seed_programs.py`.

To run a specific example:

```bash
python3 examples/seed_programs.py <example_name>
```

To run all examples:

```bash
python3 examples/seed_programs.py all
```

### Generating Visualizations

The `visualize.py` script runs a ChaosBF program and generates plots of its internal state over time.

```bash
python3 src/visualize.py "<code>" [options]
```

**Example:**

```bash
python3 src/visualize.py "++[>+<-].:{;}{?}^*=@=.#%" --steps 2000 --output output/my_run
```

This will generate the following files:
- `output/my_run_thermodynamics.png`
- `output/my_run_evolution.png`
- `output/my_run_criticality.png`
- `output/my_run_data.csv` (raw trace data)

## Instruction Set

The following table details the full ChaosBF instruction set.

| Op  | Meaning                                          | ΔE Cost        | Notes                                          |
|:----|:-------------------------------------------------|:---------------|:-----------------------------------------------|
| `>` | `ptr++`                                          | -1             | Move tape pointer right                        |
| `<` | `ptr--`                                          | -1             | Move tape pointer left                         |
| `+` | `M[p]++`                                         | -2             | Increment cell value (raises local potential)  |
| `-` | `M[p]--`                                         | +1             | Decrement cell value (dissipative release)     |
| `[` | `while M[p] != 0`                                | -(1+depth/3)   | Loop start (leak grows with nesting)           |
| `]` | `end while`                                      | -(1+depth/3)   | Loop end                                       |
| `.` | `emit M[p] to O`                                 | -1             | Output byte to buffer                          |
| `,` | `read byte -> M[p]`                              | -1             | Read byte from input (simplified to 0)         |
| `^` | `T += τ`                                         | -1             | Anneal up (increase temperature, more mutation)|
| `v` | `T -= τ`                                         | -1             | Anneal down (decrease temperature, consolidate)|
| `:` | Sense local entropy `H(M[p-w:p+w])`               | 0              | Measures entropy in a window around the pointer|
| `;` | Accrue local entropy to global `S`               | `-S_local`     | Pushes local entropy to the global state       |
| `?` | Thermal mutate one random instruction            | -2             | `P(mutation) = σ(T)`                           |
| `*` | Replicate current genome to `G`                  | -10            | Allowed if `F > θ_rep`                         |
| `@` | Crossover with random gene from `G`              | -6             | Raises population diversity                    |
| `=` | Learn: peephole optimization (`+-` -> `''`)      | `+ΔF_gain`     | Simplifies code and gains energy               |
| `!` | Select: promote to elite set if score is high    | -1             | Darwinian selection mechanism                  |
| `{` | Start critical loop                              | -2 / iter      | Enforces branching factor λ≈1                  |
| `}` | End critical loop                                | 0              |                                                |
| `#` | Measure: print `E,T,S,F,λ`                       | 0              | Provides observability into the system state   |
| `%` | Phenotype dump: print `O` then clear             | -1             | Exposes the program's behavior                 |
| `~` | Quench: revert to elite ancestor if `F < 0`      | +5             | Survival fallback mechanism                    |

## Further Exploration

This implementation provides a foundation for exploring complex adaptive systems, artificial life, and the principles of open-ended evolution. Potential areas for further research include:

- **Population Dynamics:** Running multiple ChaosBF instances in parallel and allowing them to interact.
- **Task-Specific Fitness:** Introducing specific tasks or environments to guide the evolution towards particular goals.
- **Advanced Learning:** Implementing more sophisticated learning algorithms beyond peephole optimization.
- **Visualization:** Creating more advanced visualizations of the tape, population, and evolutionary history.

