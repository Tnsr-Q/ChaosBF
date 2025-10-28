use crate::state::SimState;
use crate::rng::Rng;

/// Configuration for a single island
#[derive(Clone, Debug)]
pub struct IslandConfig {
    pub id: usize,
    pub e0: f32,          // Initial energy
    pub t0: f32,          // Initial temperature
    pub theta_rep: f32,   // Replication threshold
    pub name: String,
}

/// Elite migrating between islands
#[derive(Clone)]
pub struct Migrant {
    pub genome: Vec<u8>,
    pub fitness: f32,
    pub source_island: usize,
    pub descriptors: [f32; 4],
}

/// Single island in the ecology
pub struct Island {
    pub config: IslandConfig,
    pub population: Vec<SimState>,
    pub generation: u32,
    pub immigrants: u32,
    pub emigrants: u32,
    pub best_fitness: f32,
    pub diversity: f32,
    rng: Rng,
}

impl Island {
    pub fn new(config: IslandConfig, seed_genomes: &[Vec<u8>], seed: u64) -> Self {
        let rng = Rng::from_seed(seed + config.id as u64);
        let mut population = Vec::new();

        for genome in seed_genomes {
            let state = SimState::new(
                seed + config.id as u64 + population.len() as u64,
                256,  // width
                256,  // height
                genome.clone(),
            );
            population.push(state);
        }

        Self {
            config,
            population,
            generation: 0,
            immigrants: 0,
            emigrants: 0,
            best_fitness: 0.0,
            diversity: 0.0,
            rng,
        }
    }

    /// Evolve island population
    pub fn evolve(&mut self, steps: u32) {
        // Run all organisms
        for state in &mut self.population {
            // Run simulation steps
            for _ in 0..steps {
                if state.e <= 0.0 || state.code_len == 0 {
                    break;
                }
                state.step();
            }
        }

        self.generation += 1;
        self.update_stats();
    }

    fn update_stats(&mut self) {
        let fitnesses: Vec<f32> = self.population.iter()
            .map(|s| s.e / (s.steps as f32 + 1.0))
            .collect();

        if !fitnesses.is_empty() {
            self.best_fitness = fitnesses.iter().copied().fold(f32::NEG_INFINITY, f32::max);

            // Diversity = std dev of fitnesses
            let mean = fitnesses.iter().sum::<f32>() / fitnesses.len() as f32;
            let variance = fitnesses.iter()
                .map(|f| (f - mean).powi(2))
                .sum::<f32>() / fitnesses.len() as f32;
            self.diversity = variance.sqrt();
        }
    }

    /// Get top elites for migration
    pub fn get_elites(&self, n: usize) -> Vec<Migrant> {
        let mut candidates: Vec<(f32, &SimState)> = self.population.iter()
            .map(|s| (s.e / (s.steps as f32 + 1.0), s))
            .collect();

        candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        candidates.iter()
            .take(n)
            .map(|(fitness, state)| Migrant {
                genome: state.code[..state.code_len].to_vec(),
                fitness: *fitness,
                source_island: self.config.id,
                descriptors: [
                    state.lambda_hat,
                    state.s,
                    state.f,
                    state.complexity_estimate,
                ],
            })
            .collect()
    }

    /// Accept immigrant from another island
    pub fn accept_immigrant(&mut self, migrant: &Migrant) {
        // Create new state with island's conditions
        let mut new_state = SimState::new(
            self.rng.next_u64(),
            256,
            256,
            migrant.genome.clone(),
        );
        new_state.e = self.config.e0;
        new_state.t = self.config.t0;

        // Replace weakest individual
        if !self.population.is_empty() {
            let fitnesses: Vec<f32> = self.population.iter()
                .map(|s| s.e / (s.steps as f32 + 1.0))
                .collect();

            let weakest_idx = fitnesses.iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            self.population[weakest_idx] = new_state;
        } else {
            self.population.push(new_state);
        }

        self.immigrants += 1;
    }

    /// Compute novelty deficit relative to global descriptors
    pub fn compute_novelty_deficit(&self, global_descriptors: &[[f32; 4]]) -> f32 {
        if self.population.is_empty() || global_descriptors.is_empty() {
            return 0.0;
        }

        // Get local descriptors
        let local_descriptors: Vec<[f32; 4]> = self.population.iter()
            .map(|s| [
                s.lambda_hat,
                s.s,
                s.f,
                s.complexity_estimate,
            ])
            .collect();

        // Compute average distance to global population
        let mut total_dist = 0.0;
        let mut count = 0;

        for local in &local_descriptors {
            for global in global_descriptors {
                let sq_dist = (0..4).map(|i| (local[i] - global[i]).powi(2))
                    .sum::<f32>();
                total_dist += sq_dist;
                count += 1;
            }
        }

        let avg_sq_dist = if count > 0 { total_dist / count as f32 } else { 0.0 };
        let avg_dist = avg_sq_dist.sqrt();

        // Deficit = inverse of novelty (low distance = high deficit)
        1.0 / (avg_dist + 0.1)
    }
}

/// Multi-island ecology
pub struct IslandEcology {
    pub islands: Vec<Island>,
    pub generation: u32,
    rng: Rng,
}

impl IslandEcology {
    pub fn new(n_islands: usize, seed_genomes: &[Vec<u8>], seed: u64) -> Self {
        let rng = Rng::from_seed(seed);
        let configs = Self::create_island_configs(n_islands);

        let islands = configs.into_iter()
            .map(|config| Island::new(config, seed_genomes, seed))
            .collect();

        Self {
            islands,
            generation: 0,
            rng,
        }
    }

    fn create_island_configs(n: usize) -> Vec<IslandConfig> {
        let names = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta"];

        (0..n).map(|i| {
            let t = i as f32 / n.max(1) as f32;
            IslandConfig {
                id: i,
                e0: 150.0 + 100.0 * t,
                t0: 0.3 + 0.4 * t,
                theta_rep: 5.0 + 10.0 * t,
                name: if i < names.len() {
                    names[i].to_string()
                } else {
                    format!("Island-{}", i)
                },
            }
        }).collect()
    }

    /// Evolve all islands with periodic migration
    pub fn evolve(&mut self, steps: u32, migration_interval: u32) {
        // Evolve all islands
        for island in &mut self.islands {
            island.evolve(steps);
        }

        self.generation += 1;

        // Periodic migration
        if self.generation % migration_interval == 0 {
            self.migrate();
        }
    }

    fn migrate(&mut self) {
        // Collect all descriptors
        let global_descriptors: Vec<[f32; 4]> = self.islands.iter()
            .flat_map(|island| {
                island.population.iter().map(|s| [
                    s.lambda_hat,
                    s.s,
                    s.f,
                    s.complexity_estimate,
                ])
            })
            .collect();

        // Compute novelty deficits
        let deficits: Vec<f32> = self.islands.iter()
            .map(|island| island.compute_novelty_deficit(&global_descriptors))
            .collect();

        let total_deficit: f32 = deficits.iter().sum();

        // Normalize to probabilities
        let migration_probs: Vec<f32> = if total_deficit > 0.0 {
            deficits.iter().map(|d| d / total_deficit).collect()
        } else {
            vec![1.0 / self.islands.len() as f32; self.islands.len()]
        };

        // Each island sends migrants
        let migrations: Vec<(usize, Vec<Migrant>)> = self.islands.iter()
            .map(|island| (island.config.id, island.get_elites(2)))
            .collect();

        // Apply migrations
        for (source_id, migrants) in migrations {
            for migrant in migrants {
                // Select destination based on novelty deficit
                let mut rnd = self.rng.gen_f32();
                let mut dest_idx = 0;

                for (i, prob) in migration_probs.iter().enumerate() {
                    rnd -= prob;
                    if rnd <= 0.0 {
                        dest_idx = i;
                        break;
                    }
                }

                // Don't migrate to self
                if dest_idx != source_id && dest_idx < self.islands.len() {
                    self.islands[dest_idx].accept_immigrant(&migrant);
                    if source_id < self.islands.len() {
                        self.islands[source_id].emigrants += 1;
                    }
                }
            }
        }
    }

    pub fn total_population(&self) -> usize {
        self.islands.iter().map(|i| i.population.len()).sum()
    }

    pub fn total_migrants(&self) -> (u32, u32) {
        let immigrants = self.islands.iter().map(|i| i.immigrants).sum();
        let emigrants = self.islands.iter().map(|i| i.emigrants).sum();
        (immigrants, emigrants)
    }
}
