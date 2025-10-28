use crate::lyapunov::{TaggedElite, EdgeTag};
use crate::rng::Rng;

/// Routes emitters to spawn offspring around marginal elites
pub struct EdgeBandRouter {
    marginal_weight: f32,
    rng: Rng,

    // Statistics
    pub critical_stable: u32,
    pub critical_chaotic: u32,
    pub marginal: u32,
}

impl EdgeBandRouter {
    pub fn new(marginal_weight: f32, seed: u64) -> Self {
        Self {
            marginal_weight,
            rng: Rng::from_seed(seed),
            critical_stable: 0,
            critical_chaotic: 0,
            marginal: 0,
        }
    }

    /// Add elite and tag it based on Lyapunov CI
    pub fn add_elite(
        &mut self,
        genome: Vec<u8>,
        fitness: f32,
        lyapunov: f32,
        ci_low: f32,
        ci_high: f32,
        descriptors: [f32; 4],
    ) -> TaggedElite {
        let tag = EdgeTag::from_lyapunov_ci(ci_low, ci_high);

        // Update statistics
        match tag {
            EdgeTag::CriticalStable => self.critical_stable += 1,
            EdgeTag::CriticalChaotic => self.critical_chaotic += 1,
            EdgeTag::Marginal => self.marginal += 1,
        }

        TaggedElite {
            genome,
            fitness,
            lyapunov,
            ci_low,
            ci_high,
            tag,
            descriptors,
        }
    }

    /// Select parent elite with marginal weighting
    pub fn select_parent<'a>(&mut self, elites: &'a [TaggedElite], strategy: EdgeBandStrategy) -> Option<&'a TaggedElite> {
        if elites.is_empty() {
            return None;
        }

        match strategy {
            EdgeBandStrategy::MarginalOnly => {
                // Only select marginal elites
                let marginal: Vec<&TaggedElite> = elites.iter()
                    .filter(|e| e.tag == EdgeTag::Marginal)
                    .collect();

                if marginal.is_empty() {
                    // Fallback to random
                    Some(&elites[self.rng.gen_range(0, elites.len())])
                } else {
                    Some(marginal[self.rng.gen_range(0, marginal.len())])
                }
            }

            EdgeBandStrategy::MarginalWeighted => {
                // Weight marginal elites higher
                let weights: Vec<f32> = elites.iter()
                    .map(|e| if e.tag == EdgeTag::Marginal {
                        self.marginal_weight
                    } else {
                        1.0
                    })
                    .collect();

                let total: f32 = weights.iter().sum();
                let mut rnd = self.rng.gen_f32() * total;

                for (i, weight) in weights.iter().enumerate() {
                    rnd -= weight;
                    if rnd <= 0.0 {
                        return Some(&elites[i]);
                    }
                }

                Some(&elites[elites.len() - 1])
            }

            EdgeBandStrategy::Uniform => {
                Some(&elites[self.rng.gen_range(0, elites.len())])
            }
        }
    }

    /// Get statistics
    pub fn get_stats(&self) -> EdgeBandStats {
        let total = self.critical_stable + self.critical_chaotic + self.marginal;
        EdgeBandStats {
            critical_stable: self.critical_stable,
            critical_chaotic: self.critical_chaotic,
            marginal: self.marginal,
            total,
            marginal_fraction: if total > 0 {
                self.marginal as f32 / total as f32
            } else {
                0.0
            },
        }
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.critical_stable = 0;
        self.critical_chaotic = 0;
        self.marginal = 0;
    }
}

#[derive(Clone, Copy, Debug)]
pub enum EdgeBandStrategy {
    MarginalOnly,     // Only select marginal elites
    MarginalWeighted, // Weight marginal elites higher
    Uniform,          // Uniform selection
}

#[derive(Clone, Copy, Debug)]
pub struct EdgeBandStats {
    pub critical_stable: u32,
    pub critical_chaotic: u32,
    pub marginal: u32,
    pub total: u32,
    pub marginal_fraction: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edge_band_stats() {
        let mut router = EdgeBandRouter::new(2.0, 42);

        router.add_elite(vec![1, 2, 3], 1.0, 0.1, 0.05, 0.15, [0.0; 4]);
        router.add_elite(vec![4, 5, 6], 1.0, -0.1, -0.15, -0.05, [0.0; 4]);
        router.add_elite(vec![7, 8, 9], 1.0, 0.0, -0.05, 0.05, [0.0; 4]);

        let stats = router.get_stats();
        assert_eq!(stats.critical_chaotic, 1);
        assert_eq!(stats.critical_stable, 1);
        assert_eq!(stats.marginal, 1);
        assert_eq!(stats.total, 3);
    }
}
