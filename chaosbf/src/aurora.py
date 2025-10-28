#!/usr/bin/env python3
"""
AURORA: Learned Behavior Descriptors for ChaosBF

Trains a tiny autoencoder on phenotype traces + state summaries to learn
2 latent behavioral dimensions. These unlock niches that hand-crafted
descriptors cannot see.

Reference: "AURORA: Autonomous Robotic Unsupervised Reinforcement Learning via Autoencoders"
"""

import sys
sys.path.insert(0, '/home/ubuntu/chaosbf/src')
from chaosbf_v3 import ChaosBFv3, K
import numpy as np
from typing import List, Tuple, Dict, Optional
import json


class TinyAutoencoder:
    """
    Minimal autoencoder for learning behavioral descriptors.
    
    Architecture:
    - Input: Flattened phenotype trace + state summary
    - Encoder: input -> hidden -> latent (2D)
    - Decoder: latent -> hidden -> output
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 2,
        hidden_dim: int = 32,
        learning_rate: float = 0.01,
        seed: Optional[int] = None
    ):
        """Initialize autoencoder."""
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.lr = learning_rate
        
        # Set random seed
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize weights (Xavier initialization)
        self.W_enc1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b_enc1 = np.zeros(hidden_dim)
        
        self.W_enc2 = np.random.randn(hidden_dim, latent_dim) * np.sqrt(2.0 / hidden_dim)
        self.b_enc2 = np.zeros(latent_dim)
        
        self.W_dec1 = np.random.randn(latent_dim, hidden_dim) * np.sqrt(2.0 / latent_dim)
        self.b_dec1 = np.zeros(hidden_dim)
        
        self.W_dec2 = np.random.randn(hidden_dim, input_dim) * np.sqrt(2.0 / hidden_dim)
        self.b_dec2 = np.zeros(input_dim)
        
        # Training history
        self.losses = []
    
    def relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation."""
        return np.maximum(0, x)
    
    def relu_grad(self, x: np.ndarray) -> np.ndarray:
        """ReLU gradient."""
        return (x > 0).astype(float)
    
    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encode input to latent space."""
        h1 = self.relu(x @ self.W_enc1 + self.b_enc1)
        z = h1 @ self.W_enc2 + self.b_enc2
        return z
    
    def decode(self, z: np.ndarray) -> np.ndarray:
        """Decode latent to output."""
        h1 = self.relu(z @ self.W_dec1 + self.b_dec1)
        x_recon = h1 @ self.W_dec2 + self.b_dec2
        return x_recon
    
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass."""
        z = self.encode(x)
        x_recon = self.decode(z)
        return z, x_recon
    
    def train_step(self, x: np.ndarray) -> float:
        """Single training step with backpropagation."""
        # Forward pass
        h_enc1 = self.relu(x @ self.W_enc1 + self.b_enc1)
        z = h_enc1 @ self.W_enc2 + self.b_enc2
        h_dec1 = self.relu(z @ self.W_dec1 + self.b_dec1)
        x_recon = h_dec1 @ self.W_dec2 + self.b_dec2
        
        # Compute loss (MSE)
        loss = np.mean((x - x_recon) ** 2)
        
        # Backward pass
        # Output layer
        d_x_recon = 2 * (x_recon - x) / x.size
        d_W_dec2 = h_dec1.T @ d_x_recon
        d_b_dec2 = np.sum(d_x_recon, axis=0)
        
        # Decoder hidden layer
        d_h_dec1 = d_x_recon @ self.W_dec2.T
        d_h_dec1 *= self.relu_grad(h_dec1)
        d_W_dec1 = z.T @ d_h_dec1
        d_b_dec1 = np.sum(d_h_dec1, axis=0)
        
        # Latent layer
        d_z = d_h_dec1 @ self.W_dec1.T
        d_W_enc2 = h_enc1.T @ d_z
        d_b_enc2 = np.sum(d_z, axis=0)
        
        # Encoder hidden layer
        d_h_enc1 = d_z @ self.W_enc2.T
        d_h_enc1 *= self.relu_grad(h_enc1)
        d_W_enc1 = x.T @ d_h_enc1
        d_b_enc1 = np.sum(d_h_enc1, axis=0)
        
        # Update weights
        self.W_dec2 -= self.lr * d_W_dec2
        self.b_dec2 -= self.lr * d_b_dec2
        self.W_dec1 -= self.lr * d_W_dec1
        self.b_dec1 -= self.lr * d_b_dec1
        self.W_enc2 -= self.lr * d_W_enc2
        self.b_enc2 -= self.lr * d_b_enc2
        self.W_enc1 -= self.lr * d_W_enc1
        self.b_enc1 -= self.lr * d_b_enc1
        
        return loss
    
    def contrastive_loss(self, z_anchor: np.ndarray, z_positive: np.ndarray, 
                        z_negatives: List[np.ndarray], temperature: float = 0.1) -> float:
        """
        InfoNCE contrastive loss for temporal pairs (NumPy implementation).
        
        Args:
            z_anchor: Anchor latent embedding
            z_positive: Positive (temporal neighbor) embedding
            z_negatives: List of negative embeddings
            temperature: Temperature parameter
        
        Returns:
            Contrastive loss value
        """
        # Cosine similarity
        def cosine_sim(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
        
        pos_sim = cosine_sim(z_anchor, z_positive) / temperature
        
        neg_sims = []
        for z_neg in z_negatives:
            neg_sim = cosine_sim(z_anchor, z_neg) / temperature
            neg_sims.append(neg_sim)
        
        # InfoNCE: -log(exp(pos) / (exp(pos) + sum(exp(neg))))
        numerator = np.exp(pos_sim)
        denominator = numerator + sum(np.exp(s) for s in neg_sims)
        
        loss = -np.log(numerator / (denominator + 1e-8))
        return loss
    
    def fit(self, X: np.ndarray, epochs: int = 100, batch_size: int = 32, verbose: bool = True,
           use_contrastive: bool = True, contrastive_weight: float = 0.1):
        """
        Train autoencoder on data with optional temporal contrastive learning.
        
        Args:
            X: Training data
            epochs: Number of training epochs
            batch_size: Batch size
            verbose: Print progress
            use_contrastive: Use temporal contrastive learning
            contrastive_weight: Weight for contrastive loss
        """
        n_samples = X.shape[0]
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            
            epoch_loss = 0.0
            n_batches = 0
            
            # Mini-batch training with temporal contrastive learning
            for i in range(0, n_samples, batch_size):
                batch = X_shuffled[i:i+batch_size]
                
                # Standard reconstruction loss
                recon_loss = self.train_step(batch)
                total_loss = recon_loss
                
                # Add temporal contrastive loss
                if use_contrastive and i < n_samples - batch_size:
                    # Get temporal neighbors (next batch)
                    batch_next = X_shuffled[i+batch_size:i+2*batch_size]
                    if len(batch_next) > 0:
                        # Encode current and next batches
                        z_current = self.encode(batch)
                        z_next = self.encode(batch_next)
                        
                        # For each anchor, use next as positive
                        for j in range(min(len(z_current), len(z_next))):
                            # Negatives: random samples from other batches
                            neg_indices = np.random.choice(
                                [k for k in range(n_samples) if k < i or k >= i+2*batch_size],
                                size=min(5, max(1, n_samples - 2*batch_size)),
                                replace=False
                            )
                            z_negatives = [self.encode(X_shuffled[k:k+1])[0] for k in neg_indices]
                            
                            if z_negatives:
                                contrast_loss = self.contrastive_loss(
                                    z_current[j], z_next[j], z_negatives
                                )
                                total_loss += contrastive_weight * contrast_loss
                
                epoch_loss += total_loss
                n_batches += 1
            
            avg_loss = epoch_loss / n_batches
            self.losses.append(avg_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.6f}")
    
    def save(self, filepath: str):
        """Save model weights."""
        np.savez(
            filepath,
            W_enc1=self.W_enc1, b_enc1=self.b_enc1,
            W_enc2=self.W_enc2, b_enc2=self.b_enc2,
            W_dec1=self.W_dec1, b_dec1=self.b_dec1,
            W_dec2=self.W_dec2, b_dec2=self.b_dec2,
            losses=np.array(self.losses)
        )
        print(f"Saved model to {filepath}")
    
    def load(self, filepath: str):
        """Load model weights."""
        data = np.load(filepath)
        self.W_enc1 = data['W_enc1']
        self.b_enc1 = data['b_enc1']
        self.W_enc2 = data['W_enc2']
        self.b_enc2 = data['b_enc2']
        self.W_dec1 = data['W_dec1']
        self.b_dec1 = data['b_dec1']
        self.W_dec2 = data['W_dec2']
        self.b_dec2 = data['b_dec2']
        self.losses = data['losses'].tolist()
        print(f"Loaded model from {filepath}")


class AURORADescriptors:
    """
    AURORA-style learned behavioral descriptors for ChaosBF.
    
    Extracts phenotype traces + state summaries, trains an autoencoder,
    and uses the 2D latent space as behavioral descriptors.
    """
    
    def __init__(
        self,
        trace_length: int = 50,
        state_features: int = 10,
        latent_dim: int = 2,
        seed: Optional[int] = None
    ):
        """Initialize AURORA descriptor system."""
        self.trace_length = trace_length
        self.state_features = state_features
        self.input_dim = trace_length + state_features
        self.latent_dim = latent_dim
        self.seed = seed
        
        # Initialize autoencoder
        self.ae = TinyAutoencoder(
            input_dim=self.input_dim,
            latent_dim=latent_dim,
            hidden_dim=32,
            learning_rate=0.01,
            seed=seed
        )
        
        self.is_trained = False
        
        # Coverage tracking
        self.latent_samples = []  # Store latent coordinates for coverage analysis
    
    def extract_features(self, cbf: ChaosBFv3) -> np.ndarray:
        """
        Extract features from a ChaosBF run.
        
        Features:
        - Phenotype trace (last N bytes of output, normalized)
        - State summary (E, T, S, λ, volatility, etc.)
        """
        # Phenotype trace (last trace_length bytes of output)
        output_bytes = cbf.O.encode('latin1', errors='ignore')
        trace = np.zeros(self.trace_length)
        
        if len(output_bytes) > 0:
            for i, byte in enumerate(output_bytes[-self.trace_length:]):
                trace[i] = byte / 255.0  # Normalize to [0, 1]
        
        # State summary
        stats = cbf.get_stats()
        state = np.array([
            stats['energy'] / 200.0,  # Normalize by typical E0
            stats['temperature'],
            stats['entropy'] / 10.0,  # Normalize by typical max
            stats['branching_factor'],
            stats['lambda_volatility'] * 10.0,  # Scale up small values
            stats['output_complexity'] / 100.0,
            stats['mutations'] / 100.0,
            stats['replications'] / 10.0,
            stats['dS_dt_ema'],
            stats['dK_dt_ema']
        ])
        
        # Concatenate
        features = np.concatenate([trace, state])
        return features
    
    def collect_training_data(
        self,
        genomes: List[str],
        E: float = 200.0,
        T: float = 0.5,
        steps: int = 2000,
        verbose: bool = True
    ) -> np.ndarray:
        """Collect training data by running genomes."""
        if verbose:
            print(f"Collecting training data from {len(genomes)} genomes...")
        
        X = []
        
        for i, genome in enumerate(genomes):
            if verbose and (i + 1) % 10 == 0:
                print(f"  Evaluating genome {i+1}/{len(genomes)}...")
            
            cbf = ChaosBFv3(
                genome,
                E=E,
                T=T,
                seed=self.seed + i if self.seed else None,
                use_pid=True,
                use_variance_shaping=True,
                verbose=False
            )
            
            cbf.run(steps=steps)
            features = self.extract_features(cbf)
            X.append(features)
        
        X = np.array(X)
        
        if verbose:
            print(f"Collected {X.shape[0]} samples with {X.shape[1]} features")
        
        return X
    
    def train(
        self,
        genomes: List[str],
        E: float = 200.0,
        T: float = 0.5,
        steps: int = 2000,
        epochs: int = 100,
        batch_size: int = 32,
        verbose: bool = True
    ):
        """Train the autoencoder on genome data."""
        # Collect training data
        X = self.collect_training_data(genomes, E, T, steps, verbose)
        
        # Train autoencoder
        if verbose:
            print(f"\nTraining autoencoder for {epochs} epochs...")
        
        self.ae.fit(X, epochs=epochs, batch_size=batch_size, verbose=verbose)
        self.is_trained = True
        
        if verbose:
            print(f"Training complete. Final loss: {self.ae.losses[-1]:.6f}")
    
    def compute_descriptors(self, cbf: ChaosBFv3, track_coverage: bool = True) -> Tuple[float, float]:
        """
        Compute learned behavioral descriptors for a ChaosBF run.
        
        Args:
            cbf: ChaosBF instance
            track_coverage: Whether to track latent samples for coverage analysis
        
        Returns:
            (latent_dim_1, latent_dim_2)
        """
        if not self.is_trained:
            raise ValueError("Autoencoder not trained. Call train() first.")
        
        features = self.extract_features(cbf)
        latent = self.ae.encode(features.reshape(1, -1))
        
        # Track for coverage analysis
        if track_coverage:
            self.latent_samples.append(latent[0].copy())
        
        return float(latent[0, 0]), float(latent[0, 1])
    
    def compute_latent_coverage_kl(self, n_bins: int = 20) -> float:
        """
        Compute KL divergence between latent occupancy and uniform distribution.
        
        Lower KL = better coverage of latent space.
        
        Args:
            n_bins: Number of bins per dimension for histogram
        
        Returns:
            KL divergence
        """
        if len(self.latent_samples) < 10:
            return float('inf')
        
        latents = np.array(self.latent_samples)
        
        # Create 2D histogram
        hist, x_edges, y_edges = np.histogram2d(
            latents[:, 0], latents[:, 1],
            bins=n_bins,
            density=True
        )
        
        # Normalize to get empirical distribution
        p_empirical = hist.flatten()
        p_empirical = p_empirical / np.sum(p_empirical)
        
        # Uniform distribution
        p_uniform = np.ones_like(p_empirical) / len(p_empirical)
        
        # Compute KL divergence
        def kl_div(p, q, eps=1e-10):
            p = np.maximum(p, eps)
            q = np.maximum(q, eps)
            return float(np.sum(p * np.log(p / q)))
        
        kl = kl_div(p_empirical, p_uniform)
        
        return kl
    
    def plot_latent_coverage(self, output_path: str, n_bins: int = 20):
        """
        Plot latent space coverage heatmap.
        
        Args:
            output_path: Path to save plot
            n_bins: Number of bins per dimension
        """
        if len(self.latent_samples) < 10:
            print("Not enough samples for coverage plot")
            return
        
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
        
        latents = np.array(self.latent_samples)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('AURORA Latent Space Coverage', fontsize=14, fontweight='bold')
        
        # Heatmap
        hist, x_edges, y_edges = np.histogram2d(
            latents[:, 0], latents[:, 1],
            bins=n_bins
        )
        
        im = axes[0].imshow(
            hist.T, origin='lower',
            extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
            cmap='hot', aspect='auto'
        )
        axes[0].set_xlabel('Latent Dimension 1')
        axes[0].set_ylabel('Latent Dimension 2')
        axes[0].set_title('Occupancy Heatmap')
        plt.colorbar(im, ax=axes[0], label='Count')
        
        # Scatter plot
        axes[1].scatter(latents[:, 0], latents[:, 1], alpha=0.3, s=10)
        axes[1].set_xlabel('Latent Dimension 1')
        axes[1].set_ylabel('Latent Dimension 2')
        axes[1].set_title(f'Scatter Plot (n={len(latents)})')
        axes[1].grid(True, alpha=0.3)
        
        # Add KL divergence
        kl = self.compute_latent_coverage_kl(n_bins)
        fig.text(0.5, 0.02, f'KL(empirical || uniform) = {kl:.4f}', 
                ha='center', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved latent coverage plot to {output_path}")
        plt.close()
    
    def save(self, filepath: str):
        """Save AURORA system."""
        self.ae.save(filepath)
        
        # Save metadata
        metadata = {
            'trace_length': self.trace_length,
            'state_features': self.state_features,
            'latent_dim': self.latent_dim,
            'is_trained': self.is_trained
        }
        
        with open(filepath.replace('.npz', '_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load(self, filepath: str):
        """Load AURORA system."""
        self.ae.load(filepath)
        
        # Load metadata
        with open(filepath.replace('.npz', '_metadata.json'), 'r') as f:
            metadata = json.load(f)
        
        self.trace_length = metadata['trace_length']
        self.state_features = metadata['state_features']
        self.latent_dim = metadata['latent_dim']
        self.is_trained = metadata['is_trained']


def main():
    """Train AURORA descriptors."""
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='AURORA Learned Descriptors for ChaosBF')
    parser.add_argument('--genomes', nargs='+', help='Genomes to train on')
    parser.add_argument('--genome-file', help='File with one genome per line')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--steps', type=int, default=2000, help='Steps per evaluation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', default='/home/ubuntu/chaosbf/output/aurora_model.npz',
                       help='Output model file')
    
    args = parser.parse_args()
    
    # Get genomes
    if args.genome_file:
        with open(args.genome_file, 'r') as f:
            genomes = [line.strip() for line in f if line.strip()]
    elif args.genomes:
        genomes = args.genomes
    else:
        # Default seed genomes
        genomes = [
            '++[>+<-].:{;}{?}^*=@=.#%',
            ':+[>+<-];.#',
            '{?}{?}{?}^=v=.#',
            '*=@=:{;}#%',
            '+++[>+++<-]>{?}*=@.#%',
            '++[>+<-]^v^v:{;}#.%',
            '{+}{?}^{-}{?}v*=@:{;}#%',
            '++[>++<--].#',
            '{+}*@{-}=.#',
            ':;:;{?}^v.#'
        ]
    
    print(f"Training AURORA on {len(genomes)} genomes")
    
    # Initialize AURORA
    aurora = AURORADescriptors(
        trace_length=50,
        state_features=10,
        latent_dim=2,
        seed=args.seed
    )
    
    # Train
    aurora.train(
        genomes=genomes,
        steps=args.steps,
        epochs=args.epochs,
        verbose=True
    )
    
    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    aurora.save(args.output)
    
    print(f"\nAURORA model saved to {args.output}")


if __name__ == '__main__':
    main()

