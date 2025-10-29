#![allow(dead_code)]

use std::collections::HashMap;

/// Phenotype critic that learns to predict patterns
pub struct PhenotypeCritic {
    ngram_size: usize,
    learning_rate: f32,
    surprise_weight: f32,

    // N-gram model: context -> token -> count
    ngram_counts: HashMap<Vec<u8>, HashMap<u8, u32>>,
    total_observations: usize,
}

impl PhenotypeCritic {
    pub fn new(ngram_size: usize, learning_rate: f32, surprise_weight: f32) -> Self {
        Self {
            ngram_size,
            learning_rate,
            surprise_weight,
            ngram_counts: HashMap::new(),
            total_observations: 0,
        }
    }

    /// Get n-gram context at position
    fn get_context(&self, tokens: &[u8], pos: usize) -> Vec<u8> {
        let start = pos.saturating_sub(self.ngram_size - 1);
        tokens[start..pos].to_vec()
    }

    /// Predict next token probabilities given context
    pub fn predict(&self, context: &[u8]) -> HashMap<u8, f32> {
        if let Some(counts) = self.ngram_counts.get(context) {
            let total: u32 = counts.values().sum();

            if total > 0 {
                return counts.iter()
                    .map(|(token, count)| (*token, *count as f32 / total as f32))
                    .collect();
            }
        }

        HashMap::new()
    }

    /// Compute surprise (negative log probability) of actual token
    pub fn surprise(&self, context: &[u8], actual_token: u8) -> f32 {
        let probs = self.predict(context);

        if let Some(&prob) = probs.get(&actual_token) {
            -(prob + 1e-10).ln()
        } else {
            // Maximum surprise for unseen tokens
            10.0
        }
    }

    /// Learn from observed phenotype
    pub fn learn(&mut self, text: &[u8]) {
        for i in 0..text.len() {
            let context = self.get_context(text, i);
            let token = text[i];

            // Update n-gram counts
            self.ngram_counts
                .entry(context)
                .or_insert_with(HashMap::new)
                .entry(token)
                .and_modify(|c| *c += 1)
                .or_insert(1);

            self.total_observations += 1;
        }
    }

    /// Evaluate phenotype and return surprise score
    pub fn evaluate(&self, text: &[u8]) -> f32 {
        if text.is_empty() {
            return 0.0;
        }

        let mut total_surprise = 0.0;
        let mut count = 0;

        for i in 0..text.len() {
            let context = self.get_context(text, i);
            let token = text[i];

            total_surprise += self.surprise(&context, token);
            count += 1;
        }

        if count > 0 {
            total_surprise / count as f32
        } else {
            0.0
        }
    }

    /// Compute fitness bonus for surprising the critic
    pub fn fitness_bonus(&self, text: &[u8]) -> f32 {
        let surprise_score = self.evaluate(text);
        self.surprise_weight * surprise_score
    }

    /// Get statistics
    pub fn get_stats(&self) -> CriticStats {
        let avg_tokens_per_context = if !self.ngram_counts.is_empty() {
            self.ngram_counts.values()
                .map(|counts| counts.len() as f32)
                .sum::<f32>() / self.ngram_counts.len() as f32
        } else {
            0.0
        };

        CriticStats {
            ngram_size: self.ngram_size,
            total_observations: self.total_observations,
            unique_contexts: self.ngram_counts.len(),
            avg_tokens_per_context,
        }
    }

    /// Reset critic state
    pub fn reset(&mut self) {
        self.ngram_counts.clear();
        self.total_observations = 0;
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CriticStats {
    pub ngram_size: usize,
    pub total_observations: usize,
    pub unique_contexts: usize,
    pub avg_tokens_per_context: f32,
}

/// Co-evolution system with critic feedback
pub struct CriticEvolution {
    pub critic: PhenotypeCritic,
    population_size: usize,
    generation: u32,
}

impl CriticEvolution {
    pub fn new(ngram_size: usize, surprise_weight: f32, population_size: usize) -> Self {
        Self {
            critic: PhenotypeCritic::new(ngram_size, 0.1, surprise_weight),
            population_size,
            generation: 0,
        }
    }

    /// Compute fitness with critic bonus
    pub fn compute_fitness_with_critic(&self, base_fitness: f32, output: &[u8]) -> (f32, f32) {
        let surprise_bonus = self.critic.fitness_bonus(output);
        let total_fitness = base_fitness + surprise_bonus;
        (total_fitness, surprise_bonus)
    }

    /// Learn from population outputs
    pub fn learn_from_population(&mut self, outputs: &[Vec<u8>]) {
        for output in outputs {
            self.critic.learn(output);
        }
        self.generation += 1;
    }

    pub fn get_generation(&self) -> u32 {
        self.generation
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_critic_learning() {
        let mut critic = PhenotypeCritic::new(3, 0.1, 1.0);

        // Learn from some data
        let data = b"Hello world";
        critic.learn(data);

        // Evaluate familiar vs novel
        let familiar_surprise = critic.evaluate(b"Hello");
        let novel_surprise = critic.evaluate(b"Goodbye");

        // Novel should have higher surprise
        assert!(novel_surprise > familiar_surprise);
    }

    #[test]
    fn test_critic_stats() {
        let mut critic = PhenotypeCritic::new(2, 0.1, 1.0);
        critic.learn(b"abc");

        let stats = critic.get_stats();
        assert!(stats.total_observations > 0);
        assert!(stats.unique_contexts > 0);
    }
}
