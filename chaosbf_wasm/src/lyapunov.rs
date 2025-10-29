#![allow(dead_code)]

use crate::state::SimState;

/// Lyapunov exponent estimator via twin-run divergence
pub struct LyapunovEstimator {
    perturbation: i64,
    window_size: usize,
}

impl LyapunovEstimator {
    pub fn new(perturbation: f32, window_size: usize) -> Self {
        Self {
            perturbation: (perturbation * 1e6).max(1.0) as i64,
            window_size,
        }
    }

    /// Compute descriptor divergence between two clones
    pub fn compute_descriptor_divergence(&self, state_a: &SimState, state_b: &SimState) -> f32 {
        // Extract behavioral descriptors
        let desc_a = [
            state_a.lambda_hat - 1.0,     // λ-deviation
            state_a.info_per_energy,
            state_a.ds_dt_ema,             // Entropy slope
            state_a.lambda_volatility,
        ];

        let desc_b = [
            state_b.lambda_hat - 1.0,
            state_b.info_per_energy,
            state_b.ds_dt_ema,
            state_b.lambda_volatility,
        ];

        // Euclidean distance
        desc_a.iter()
            .zip(&desc_b)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    /// Estimate Lyapunov exponent from divergence
    pub fn estimate_lyapunov(&self, divergence: f32, steps: usize) -> f32 {
        // λ_lyap ≈ log(divergence) / steps
        // Clamp to reasonable range
        let log_div = (divergence + 1e-10).ln();
        let lambda = log_div / steps.max(1) as f32;

        lambda.clamp(-1.0, 1.0)
    }

    /// Bootstrap confidence interval estimation (requires multiple runs)
    pub fn bootstrap_ci(&self, samples: &[f32], confidence: f32) -> (f32, f32, f32) {
        if samples.is_empty() {
            return (0.0, 0.0, 0.0);
        }

        let mut sorted = samples.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n = sorted.len();
        let alpha = (1.0 - confidence) / 2.0;
        let lower_idx = (n as f32 * alpha) as usize;
        let upper_idx = (n as f32 * (1.0 - alpha)) as usize;

        let mean = samples.iter().sum::<f32>() / n as f32;
        let ci_low = sorted[lower_idx.min(n - 1)];
        let ci_high = sorted[upper_idx.min(n - 1)];

        (mean, ci_low, ci_high)
    }
}

/// Tagged elite with Lyapunov classification
#[derive(Clone, Debug)]
pub struct TaggedElite {
    pub genome: Vec<u8>,
    pub fitness: f32,
    pub lyapunov: f32,
    pub ci_low: f32,
    pub ci_high: f32,
    pub tag: EdgeTag,
    pub descriptors: [f32; 4],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EdgeTag {
    CriticalStable,   // CI entirely below 0
    CriticalChaotic,  // CI entirely above 0
    Marginal,         // CI straddles 0 - true edge!
}

impl EdgeTag {
    pub fn from_lyapunov_ci(ci_low: f32, ci_high: f32) -> Self {
        if ci_high < 0.0 {
            EdgeTag::CriticalStable
        } else if ci_low > 0.0 {
            EdgeTag::CriticalChaotic
        } else {
            EdgeTag::Marginal
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            EdgeTag::CriticalStable => "critical-stable",
            EdgeTag::CriticalChaotic => "critical-chaotic",
            EdgeTag::Marginal => "marginal",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edge_tag_classification() {
        assert_eq!(EdgeTag::from_lyapunov_ci(-0.5, -0.1), EdgeTag::CriticalStable);
        assert_eq!(EdgeTag::from_lyapunov_ci(0.1, 0.5), EdgeTag::CriticalChaotic);
        assert_eq!(EdgeTag::from_lyapunov_ci(-0.1, 0.1), EdgeTag::Marginal);
    }
}
