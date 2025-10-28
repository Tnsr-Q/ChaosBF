use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use crate::state::SimState;

/// Tiny autoencoder for learning behavioral descriptors
pub struct TinyAutoencoder {
    // Encoder weights
    w_enc1: Vec<Vec<f32>>,  // input_dim x hidden_dim
    b_enc1: Vec<f32>,
    w_enc2: Vec<Vec<f32>>,  // hidden_dim x latent_dim
    b_enc2: Vec<f32>,

    // Decoder weights
    w_dec1: Vec<Vec<f32>>,  // latent_dim x hidden_dim
    b_dec1: Vec<f32>,
    w_dec2: Vec<Vec<f32>>,  // hidden_dim x input_dim
    b_dec2: Vec<f32>,

    // Dimensions
    input_dim: usize,
    hidden_dim: usize,
    latent_dim: usize,

    // Training params
    learning_rate: f32,

    // History
    pub losses: Vec<f32>,
}

impl TinyAutoencoder {
    pub fn new(input_dim: usize, latent_dim: usize, hidden_dim: usize, learning_rate: f32, seed: u64) -> Self {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);

        // Xavier initialization helper
        let xavier_init = |rows: usize, cols: usize, rng: &mut ChaCha20Rng| -> Vec<Vec<f32>> {
            let scale = (2.0 / rows as f32).sqrt();
            (0..rows).map(|_| {
                (0..cols).map(|_| rng.gen::<f32>() * 2.0 * scale - scale).collect()
            }).collect()
        };

        Self {
            w_enc1: xavier_init(input_dim, hidden_dim, &mut rng),
            b_enc1: vec![0.0; hidden_dim],
            w_enc2: xavier_init(hidden_dim, latent_dim, &mut rng),
            b_enc2: vec![0.0; latent_dim],
            w_dec1: xavier_init(latent_dim, hidden_dim, &mut rng),
            b_dec1: vec![0.0; hidden_dim],
            w_dec2: xavier_init(hidden_dim, input_dim, &mut rng),
            b_dec2: vec![0.0; input_dim],
            input_dim,
            hidden_dim,
            latent_dim,
            learning_rate,
            losses: Vec::new(),
        }
    }

    #[inline]
    fn relu(x: f32) -> f32 {
        x.max(0.0)
    }

    #[inline]
    fn relu_grad(x: f32) -> f32 {
        if x > 0.0 { 1.0 } else { 0.0 }
    }

    /// Matrix-vector multiplication
    fn mat_vec_mul(mat: &[Vec<f32>], vec: &[f32]) -> Vec<f32> {
        mat.iter().map(|row| {
            row.iter().zip(vec.iter()).map(|(a, b)| a * b).sum()
        }).collect()
    }

    /// Encode input to latent space
    pub fn encode(&self, x: &[f32]) -> Vec<f32> {
        // h1 = ReLU(x @ W_enc1 + b_enc1)
        let mut h1: Vec<f32> = Self::mat_vec_mul(&self.w_enc1, x);
        for (h, b) in h1.iter_mut().zip(&self.b_enc1) {
            *h = Self::relu(*h + b);
        }

        // z = h1 @ W_enc2 + b_enc2
        let mut z = Self::mat_vec_mul(&self.w_enc2, &h1);
        for (zval, b) in z.iter_mut().zip(&self.b_enc2) {
            *zval += b;
        }

        z
    }

    /// Decode latent to output
    pub fn decode(&self, z: &[f32]) -> Vec<f32> {
        // h1 = ReLU(z @ W_dec1 + b_dec1)
        let mut h1 = Self::mat_vec_mul(&self.w_dec1, z);
        for (h, b) in h1.iter_mut().zip(&self.b_dec1) {
            *h = Self::relu(*h + b);
        }

        // x_recon = h1 @ W_dec2 + b_dec2
        let mut x_recon = Self::mat_vec_mul(&self.w_dec2, &h1);
        for (x, b) in x_recon.iter_mut().zip(&self.b_dec2) {
            *x += b;
        }

        x_recon
    }

    /// InfoNCE contrastive loss for temporal stability
    pub fn contrastive_loss(&self, z_anchor: &[f32], z_positive: &[f32], z_negatives: &[Vec<f32>], temperature: f32) -> f32 {
        // Cosine similarity
        let cosine_sim = |a: &[f32], b: &[f32]| -> f32 {
            let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
            let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
            dot / (norm_a * norm_b + 1e-8)
        };

        let pos_sim = cosine_sim(z_anchor, z_positive) / temperature;
        let neg_sims: Vec<f32> = z_negatives.iter()
            .map(|z_neg| cosine_sim(z_anchor, z_neg.as_slice()) / temperature)
            .collect();

        // InfoNCE: -log(exp(pos) / (exp(pos) + sum(exp(neg))))
        let numerator = pos_sim.exp();
        let denominator = numerator + neg_sims.iter().map(|s| s.exp()).sum::<f32>();

        -(numerator / (denominator + 1e-8)).ln()
    }

    /// Train step with reconstruction + contrastive loss
    pub fn train_step(&mut self, x: &[f32], use_contrastive: bool, temporal_pairs: Option<(&[f32], &[Vec<f32>])>, contrastive_weight: f32) -> f32 {
        // Forward pass (simplified backprop - full implementation would be extensive)
        let z = self.encode(x);
        let x_recon = self.decode(&z);

        // MSE loss
        let mut recon_loss = 0.0;
        for (xi, xr) in x.iter().zip(&x_recon) {
            let diff = xi - xr;
            recon_loss += diff * diff;
        }
        recon_loss /= x.len() as f32;

        let mut total_loss = recon_loss;

        // Add contrastive loss if enabled
        if use_contrastive {
            if let Some((z_positive, z_negatives)) = temporal_pairs {
                let contrast_loss = self.contrastive_loss(&z, z_positive, z_negatives, 0.1);
                total_loss += contrastive_weight * contrast_loss;
            }
        }

        // TODO: Full backprop gradient descent (simplified for now)
        // In production, this would compute gradients and update all weights

        self.losses.push(total_loss);
        total_loss
    }
}

/// AURORA descriptor system
pub struct AURORADescriptors {
    autoencoder: TinyAutoencoder,
    trace_length: usize,
    state_features: usize,
    is_trained: bool,
    pub latent_samples: Vec<Vec<f32>>,
}

impl AURORADescriptors {
    pub fn new(trace_length: usize, state_features: usize, latent_dim: usize, seed: u64) -> Self {
        let input_dim = trace_length + state_features;
        Self {
            autoencoder: TinyAutoencoder::new(input_dim, latent_dim, 32, 0.01, seed),
            trace_length,
            state_features,
            is_trained: false,
            latent_samples: Vec::new(),
        }
    }

    /// Extract features from simulation state
    pub fn extract_features(&self, state: &SimState) -> Vec<f32> {
        let mut features = Vec::with_capacity(self.trace_length + self.state_features);

        // Phenotype trace (last N bytes of output, normalized)
        let output_bytes = &state.output_buffer[..state.output_len];
        let start = output_bytes.len().saturating_sub(self.trace_length);

        for i in 0..self.trace_length {
            let idx = start + i;
            if idx < output_bytes.len() {
                features.push(output_bytes[idx] as f32 / 255.0);
            } else {
                features.push(0.0);
            }
        }

        // State summary (10 features)
        features.push(state.e / 200.0);              // Normalized energy
        features.push(state.t);                       // Temperature
        features.push(state.s / 10.0);                // Normalized entropy
        features.push(state.lambda_hat);              // Branching factor
        features.push(state.lambda_volatility * 10.0); // Scaled volatility
        features.push(state.complexity_estimate / 100.0); // Output complexity
        features.push(state.mutations as f32 / 100.0);
        features.push(state.replications as f32 / 10.0);
        features.push(state.ds_dt_ema);
        features.push(state.dk_dt_ema);

        features
    }

    /// Compute learned descriptors
    pub fn compute_descriptors(&mut self, state: &SimState, track_coverage: bool) -> (f32, f32) {
        if !self.is_trained {
            // Return default descriptors if not trained
            return (0.0, 0.0);
        }

        let features = self.extract_features(state);
        let latent = self.autoencoder.encode(&features);

        if track_coverage && latent.len() >= 2 {
            self.latent_samples.push(latent.clone());
        }

        (latent.get(0).copied().unwrap_or(0.0),
         latent.get(1).copied().unwrap_or(0.0))
    }

    /// Train on collected data (simplified version)
    pub fn train(&mut self, training_data: &[Vec<f32>], epochs: usize, use_contrastive: bool, contrastive_weight: f32) {
        for epoch in 0..epochs {
            let mut epoch_loss = 0.0;
            let n_batches = training_data.len();

            for (i, sample) in training_data.iter().enumerate() {
                // Simple reconstruction training
                let loss = self.autoencoder.train_step(
                    sample,
                    use_contrastive && i < training_data.len() - 1,
                    if use_contrastive && i < training_data.len() - 1 {
                        // Temporal pair: next sample as positive
                        Some((sample.as_slice(), &vec![training_data[i + 1].clone()]))
                    } else {
                        None
                    },
                    contrastive_weight
                );
                epoch_loss += loss;
            }

            let avg_loss = epoch_loss / n_batches as f32;

            if (epoch + 1) % 10 == 0 {
                // Progress logging would go here
            }
        }

        self.is_trained = true;
    }
}
