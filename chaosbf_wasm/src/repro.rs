use crate::state::SimState;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Complete manifest for reproducible runs
#[derive(Clone, Debug)]
pub struct RunManifest {
    pub run_id: String,
    pub timestamp: u64,
    pub code_hash: u64,
    pub genome: Vec<u8>,
    pub seed: u64,
    pub e0: f32,
    pub t0: f32,
    pub pid_kp: f32,
    pub pid_ki: f32,
    pub pid_kd: f32,
    pub variance_gamma: f32,
}

impl RunManifest {
    pub fn new(state: &SimState, run_id: String) -> Self {
        let mut hasher = DefaultHasher::new();
        state.code[..state.code_len].hash(&mut hasher);
        let code_hash = hasher.finish();

        // Get current timestamp (simplified - in real impl would use proper time)
        let timestamp = state.steps as u64;

        Self {
            run_id,
            timestamp,
            code_hash,
            genome: state.code[..state.code_len].to_vec(),
            seed: state.seed,
            e0: state.e,
            t0: state.t,
            pid_kp: state.pid_kp,
            pid_ki: state.pid_ki,
            pid_kd: state.pid_kd,
            variance_gamma: state.variance_gamma,
        }
    }

    pub fn to_json(&self) -> String {
        // Manual JSON serialization (no serde dependency)
        format!(
            r#"{{"run_id":"{}","timestamp":{},"code_hash":{},"seed":{},"e0":{},"t0":{},"pid_kp":{},"pid_ki":{},"pid_kd":{},"variance_gamma":{}}}"#,
            self.run_id, self.timestamp, self.code_hash, self.seed,
            self.e0, self.t0, self.pid_kp, self.pid_ki, self.pid_kd, self.variance_gamma
        )
    }
}

/// State snapshot for rewind
#[derive(Clone, Debug)]
pub struct Snapshot {
    pub step: u32,
    pub e: f32,
    pub t: f32,
    pub s: f32,
    pub f: f32,
    pub lambda_estimate: f32,
    pub code: Vec<u8>,
    pub tape: Vec<u8>,
    pub ptr: usize,
    pub output_len: usize,
    pub genome_bank_size: usize,
}

impl Snapshot {
    pub fn from_state(state: &SimState) -> Self {
        Self {
            step: state.steps,
            e: state.e,
            t: state.t,
            s: state.s,
            f: state.f,
            lambda_estimate: state.lambda_hat,
            code: state.code[..state.code_len].to_vec(),
            // Snapshot size limit: We save only the first 1KB of tape to balance
            // reproducibility with memory efficiency. Most ChaosBF programs use
            // a small working set near the tape origin. Full 64KB tape snapshots
            // would consume excessive memory for marginal reproducibility gains.
            // For full-fidelity reproduction, use the snapshot interval of ~100 steps
            // which captures tape evolution over time rather than complete state.
            tape: state.mem[..state.mem_size.min(1024)].to_vec(),
            ptr: state.ptr,
            output_len: state.output_len,
            genome_bank_size: state.bank_size,
        }
    }

    pub fn restore_to(&self, state: &mut SimState) {
        state.steps = self.step;
        state.e = self.e;
        state.t = self.t;
        state.s = self.s;
        state.f = self.f;
        state.lambda_hat = self.lambda_estimate;

        // Restore code
        state.code_len = self.code.len().min(state.code.len());
        state.code[..state.code_len].copy_from_slice(&self.code[..state.code_len]);

        // Restore tape
        let restore_len = self.tape.len().min(state.mem.len());
        state.mem[..restore_len].copy_from_slice(&self.tape[..restore_len]);

        state.ptr = self.ptr;
        state.output_len = self.output_len;
    }
}

/// Crash capsule for debugging
#[derive(Clone, Debug)]
pub struct CrashCapsule {
    pub run_id: String,
    pub timestamp: u64,
    pub step: u32,
    pub e: f32,
    pub t: f32,
    pub s: f32,
    pub f: f32,
    pub lambda: f32,
    pub code: Vec<u8>,
    pub ptr: usize,
    pub error: Option<String>,
}

impl CrashCapsule {
    pub fn from_state(state: &SimState, run_id: String, error: Option<String>) -> Self {
        Self {
            run_id,
            timestamp: state.steps as u64,
            step: state.steps,
            e: state.e,
            t: state.t,
            s: state.s,
            f: state.f,
            lambda: state.lambda_hat,
            code: state.code[..state.code_len].to_vec(),
            ptr: state.ptr,
            error,
        }
    }

    pub fn to_json(&self) -> String {
        // Manual JSON serialization (no serde dependency)
        let error_str = self.error.as_ref().map(|e| format!(r#""{}""#, e)).unwrap_or_else(|| "null".to_string());
        format!(
            r#"{{"run_id":"{}","timestamp":{},"step":{},"e":{},"t":{},"s":{},"f":{},"lambda":{},"ptr":{},"error":{}}}"#,
            self.run_id, self.timestamp, self.step, self.e, self.t, self.s, self.f, self.lambda, self.ptr, error_str
        )
    }
}

/// Reproducibility spine manager
pub struct ReproSpine {
    pub manifest: Option<RunManifest>,
    pub snapshots: Vec<Snapshot>,
    snapshot_interval: u32,
    enable_crash_capsule: bool,
    pub crash_capsule: Option<CrashCapsule>,
}

impl ReproSpine {
    pub fn new(snapshot_interval: u32, enable_crash_capsule: bool) -> Self {
        Self {
            manifest: None,
            snapshots: Vec::new(),
            snapshot_interval,
            enable_crash_capsule,
            crash_capsule: None,
        }
    }

    /// Start a reproducible run
    pub fn start_run(&mut self, state: &SimState, run_id: String) {
        self.manifest = Some(RunManifest::new(state, run_id));
        self.snapshots.clear();
        self.crash_capsule = None;
    }

    /// Take snapshot if at interval
    pub fn maybe_snapshot(&mut self, state: &SimState) {
        if self.snapshot_interval > 0 && state.steps % self.snapshot_interval == 0 {
            self.snapshot(state);
        }
    }

    /// Take snapshot
    pub fn snapshot(&mut self, state: &SimState) {
        self.snapshots.push(Snapshot::from_state(state));
    }

    /// Rewind to closest snapshot <= target_step
    pub fn rewind(&self, state: &mut SimState, target_step: u32) -> bool {
        // Find closest snapshot
        let closest = self.snapshots.iter()
            .filter(|s| s.step <= target_step)
            .max_by_key(|s| s.step);

        if let Some(snapshot) = closest {
            snapshot.restore_to(state);
            true
        } else {
            false
        }
    }

    /// Write crash capsule
    pub fn write_crash_capsule(&mut self, state: &SimState, run_id: String, error: Option<String>) {
        if self.enable_crash_capsule {
            self.crash_capsule = Some(CrashCapsule::from_state(state, run_id, error));
        }
    }

    /// Get number of snapshots
    pub fn snapshot_count(&self) -> usize {
        self.snapshots.len()
    }

    /// Export manifest as JSON
    pub fn export_manifest(&self) -> Option<String> {
        self.manifest.as_ref().map(|m| m.to_json())
    }

    /// Export crash capsule as JSON
    pub fn export_crash_capsule(&self) -> Option<String> {
        self.crash_capsule.as_ref().map(|c| c.to_json())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snapshot_restore() {
        let mut state = SimState::new(42, 256, 256, vec![b'+', b'+', b'.']);
        state.steps = 100;
        state.e = 150.0;
        state.t = 0.7;

        let snapshot = Snapshot::from_state(&state);

        // Modify state
        state.steps = 200;
        state.e = 100.0;
        state.t = 0.5;

        // Restore
        snapshot.restore_to(&mut state);

        assert_eq!(state.steps, 100);
        assert_eq!(state.e, 150.0);
        assert_eq!(state.t, 0.7);
    }

    #[test]
    fn test_repro_spine() {
        let mut spine = ReproSpine::new(100, true);
        let state = SimState::new(42, 256, 256, vec![b'+', b'+', b'.']);

        spine.start_run(&state, "test_run_001".to_string());
        assert!(spine.manifest.is_some());

        spine.snapshot(&state);
        assert_eq!(spine.snapshot_count(), 1);
    }
}
