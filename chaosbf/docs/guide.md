# ChaosBF User Guide

This guide provides a comprehensive overview of the ChaosBF system, its theoretical underpinnings, and practical instructions for using the provided tools to run, analyze, and evolve programs.

## 1. Introduction to ChaosBF

ChaosBF is an experimental programming language designed to explore the principles of complex adaptive systems, artificial life, and open-ended evolution. It extends the minimalist syntax of Brainfuck with a rich set of features inspired by thermodynamics and evolutionary biology.

The core idea is to create a computational environment where programs are not just executed, but can also **evolve, adapt, and self-organize** under simulated physical constraints. This is achieved by treating programs as "digital organisms" that consume energy, are subject to mutation, and compete for survival in a population.

### Core Concepts

- **Thermodynamic Model:** Every action has an energy cost or gain, governed by a simulated thermodynamic system. This provides a natural constraint on program complexity and execution time.
- **Edge-of-Chaos Dynamics:** The system is designed to operate near a critical point between order and chaos, a regime believed to be fertile ground for complex computation and emergence.
- **Evolutionary Operators:** The language includes built-in operators for mutation, replication, and crossover, allowing programs to evolve over time.
- **Open-Endedness:** Unlike traditional programming, ChaosBF is not designed for solving specific, predefined problems. Instead, it fosters the emergence of novel and complex behaviors in an open-ended manner.

## 2. The Thermodynamic Core

The behavior of a ChaosBF program is deeply tied to its thermodynamic state. This state is defined by four key variables:

- **Energy (E):** The primary resource for computation. Every instruction consumes or, in some cases, produces energy. A program halts when its energy level drops to or below zero.
- **Temperature (T):** A measure of the system's volatility. Higher temperatures increase the probability of mutations, making the system more chaotic and exploratory. Lower temperatures favor stability and consolidation of existing structures.
- **Entropy (S):** A measure of the disorder or complexity of the system, both in its memory tape and its population of genomes. Certain instructions allow the system to "sense" and react to its own entropy.
- **Free Energy (F):** Defined as `F = E - T*S`, this quantity represents the energy available for doing useful work (in this case, complex computation and replication). It provides a balance between raw energy and the cost of maintaining order (low entropy).

This model creates a dynamic landscape where programs must efficiently manage their energy resources to survive and reproduce.

![Thermodynamics Plot](/home/ubuntu/chaosbf/output/demo_thermodynamics.png)
*Figure 1: Evolution of thermodynamic state variables over a sample run.* 

## 3. Evolutionary and Learning Operators

ChaosBF's most unique features are its built-in operators for evolution and learning.

### Mutation (`?`)
The `?` operator introduces random changes into the program's code. The probability of a mutation occurring is a function of the system's temperature `T`. This allows for a form of simulated annealing, where the system can explore new possibilities at high temperatures and then refine them as it cools.

### Replication (`*`)
The `*` operator allows a program to create a copy of its own code (genome) and add it to a global "gene bank" (`G`). This is a costly operation and is only permitted when the system has a high level of free energy (`F > θ_rep`), ensuring that only "fit" or energy-efficient programs can reproduce.

### Crossover (`@`)
The `@` operator takes a random genome from the gene bank and combines it with the current program's code at a single crossover point. This is a form of sexual reproduction, allowing for the rapid combination of successful traits from different evolutionary lineages.

### Learning (`=`)
The `=` operator provides a simple form of self-optimization. It scans the code for redundant patterns (e.g., `+-` or `><`) and removes them. The resulting simplification provides a direct energy gain, rewarding programs that can make themselves more efficient.

### Selection (`!`)
The `!` operator provides a mechanism for explicit Darwinian selection. It evaluates the "fitness" of the current program based on the complexity of its output (`K(O)`) relative to the energy it has consumed. Highly fit programs are promoted to an "elite" set, which can be used as a source for recovery via the `~` (quench) operator.

![Evolution Plot](/home/ubuntu/chaosbf/output/demo_evolution.png)
*Figure 2: Growth of the genome bank and evolutionary events over time.*

## 4. Criticality and the Edge of Chaos

A central theme of ChaosBF is the concept of "edge-of-chaos" dynamics. This is the idea that the most interesting and complex behaviors in a system occur in a narrow transitional zone between highly ordered (static) and highly disordered (chaotic) states.

ChaosBF attempts to maintain its operation within this zone through two mechanisms:

1.  **The `{}` Critical Loop:** This special loop structure is designed to force the system's **branching factor (λ)** towards a value of 1. The branching factor is a measure of how many different execution paths are likely to be taken. 
    - `λ < 1`: The system is sub-critical and likely to halt or become static.
    - `λ > 1`: The system is super-critical and likely to explode into chaotic, unpredictable behavior.
    - `λ ≈ 1`: The system is at criticality, allowing for the propagation of information and the emergence of complex structures.

2.  **Temperature Regulation:** The temperature `T` directly influences the branching factor (via mutation). By using the `^` (anneal up) and `v` (anneal down) operators, programs can learn to regulate their own temperature to stay near the critical point.

![Criticality Plot](/home/ubuntu/chaosbf/output/demo_criticality.png)
*Figure 3: The branching factor (λ) hovering around the critical value of 1.0.*

## 5. Practical Usage

### Running Programs

The primary way to interact with ChaosBF is through the `chaosbf.py` interpreter. 

```bash
# From the project root directory
cd /home/ubuntu/chaosbf

# Run a simple program
python3 src/chaosbf.py "++[>+<-]." --energy 100 --steps 500
```

You can control the initial conditions of the simulation with the following flags:
- `--energy E`: Set the initial energy.
- `--temp T`: Set the initial temperature.
- `--steps N`: Set the maximum number of execution steps.

### Using the Visualization Tools

The `visualize.py` script is the best way to understand the complex dynamics of a ChaosBF run. It executes a program while tracing its internal state and then generates a series of plots.

```bash
# Run the visualizer on the main seed program
python3 src/visualize.py "++[>+<-].:{;}{?}^*=@=.#%" --steps 4000 --output output/seed_run
```

This will produce:
- `output/seed_run_thermodynamics.png`
- `output/seed_run_evolution.png`
- `output/seed_run_criticality.png`
- `output/seed_run_data.csv` (a CSV file with the raw trace data for further analysis)

### Writing and Evolving Programs

There is no single "right" way to write a ChaosBF program. The most effective approach is to start with a simple "seed" and let it evolve.

Here are some strategies:

- **Start with a goal:** While the system is open-ended, you can try to create seeds that perform simple tasks, like counting or generating patterns.
- **Embrace the chaos:** Include mutation (`?`) and temperature control (`^`, `v`) operators to allow your program to explore.
- **Encourage replication:** Use the `*` and `@` operators to build a population and allow for the sharing of successful traits.
- **Observe and select:** Use the visualization tools to monitor the behavior of your programs. If you see interesting emergent behavior, you can isolate that genome and use it as the seed for a new run.

By iterating through this process of seeding, evolving, and selecting, you can guide the system towards the discovery of increasingly complex and interesting computational organisms.

