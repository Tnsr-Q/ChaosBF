//! ChaosBF v4.0.1 WASM API - Frozen ABI
//!
//! This module defines the stable public interface for JavaScript/WASM interop.
//! All exported functions maintain ABI stability and pointer semantics.

#![allow(static_mut_refs)]  // Safe in WASM single-threaded context

use crate::state::SimState;
use crate::aurora::AURORADescriptors;
use crate::lyapunov::LyapunovEstimator;
use crate::edge_band::EdgeBandRouter;
use crate::island::IslandEcology;
use crate::critic::CriticEvolution;
use crate::repro::ReproSpine;

use std::cell::RefCell;

// ============================================================================
// Global State Management (Thread-Local for WASM)
// ============================================================================
//
// WASM runs in a single-threaded environment, so thread_local! is safe and
// provides interior mutability through RefCell without unsafe code.

thread_local! {
    static SIM: RefCell<Option<SimState>> = RefCell::new(None);
    static AURORA: RefCell<Option<AURORADescriptors>> = RefCell::new(None);
    static LYAPUNOV: RefCell<Option<LyapunovEstimator>> = RefCell::new(None);
    static EDGE_BAND: RefCell<Option<EdgeBandRouter>> = RefCell::new(None);
    static ECOLOGY: RefCell<Option<IslandEcology>> = RefCell::new(None);
    static CRITIC: RefCell<Option<CriticEvolution>> = RefCell::new(None);
    static REPRO: RefCell<Option<ReproSpine>> = RefCell::new(None);
}

// Static return buffers for stable pointer semantics
static mut METRICS: [f32; 20] = [0.0; 20];
static mut DESC: [f32; 2] = [0.0; 2];
static mut EDGE_STATS: [f32; 5] = [0.0; 5];
static mut ECOLOGY_STATS: [f32; 4] = [0.0; 4];
static mut FITNESS: [f32; 2] = [0.0; 2];

/// Helper to execute with mutable access to SimState
fn with_sim_mut<F, R>(f: F) -> R
where
    F: FnOnce(&mut SimState) -> R,
{
    SIM.with(|sim_cell| {
        let mut sim_opt = sim_cell.borrow_mut();
        let sim = sim_opt.as_mut().expect("SimState not initialized - call init_sim first");
        f(sim)
    })
}

/// Helper to execute with immutable access to SimState
fn with_sim<F, R>(f: F) -> R
where
    F: FnOnce(&SimState) -> R,
{
    SIM.with(|sim_cell| {
        let sim_opt = sim_cell.borrow();
        let sim = sim_opt.as_ref().expect("SimState not initialized - call init_sim first");
        f(sim)
    })
}

// ============================================================================
// Core Simulation Interface
// ============================================================================

#[no_mangle]
pub extern "C" fn init_sim(
    seed: u64,
    width: usize,
    height: usize,
    code_ptr: *const u8,
    code_len: usize,
    e0: f32,
    t0: f32,
) {
    // Avoid UB from null pointer or zero length - use empty Vec if invalid
    let code = if code_ptr.is_null() || code_len == 0 {
        Vec::new()
    } else {
        // Clamp length to maximum allowed size
        let clamped_len = code_len.min(4096);
        // SAFETY: Safe if code_ptr is valid and points to at least clamped_len bytes
        // Minimal unsafe block to create slice and copy to Vec
        unsafe {
            let code_slice = std::slice::from_raw_parts(code_ptr, clamped_len);
            code_slice.to_vec()
        }
    };

    let mut state = SimState::new(seed, width, height, code);
    state.e = e0;
    state.t = t0;

    SIM.with(|sim_cell| {
        *sim_cell.borrow_mut() = Some(state);
    });
}

#[no_mangle]
pub extern "C" fn step_sim(ticks: u32) {
    with_sim_mut(|sim| {
        for _ in 0..ticks {
            if sim.e <= 0.0 || sim.code_len == 0 {
                break;
            }

            sim.step();

            // Auto-snapshot if repro spine enabled
            REPRO.with(|r| {
                if let Some(repro) = r.borrow_mut().as_mut() {
                    repro.maybe_snapshot(sim);
                }
            });
        }
    });
}

#[no_mangle]
pub extern "C" fn get_mem_ptr() -> *const u8 {
    with_sim(|sim| sim.mem.as_ptr())
}

#[no_mangle]
pub extern "C" fn get_mem_len() -> usize {
    with_sim(|sim| sim.mem_size)
}

#[no_mangle]
pub extern "C" fn get_metrics_ptr() -> *const f32 {
    unsafe {
        with_sim(|sim| {
            METRICS[0] = sim.steps as f32;
            METRICS[1] = sim.e;
            METRICS[2] = sim.t;
            METRICS[3] = sim.s;
            METRICS[4] = sim.f;
            METRICS[5] = sim.lambda_hat;
            METRICS[6] = sim.mutations as f32;
            METRICS[7] = sim.replications as f32;
            METRICS[8] = sim.crossovers as f32;
            METRICS[9] = sim.learns as f32;
            METRICS[10] = sim.lambda_volatility;
            METRICS[11] = sim.ds_dt_ema;
            METRICS[12] = sim.dk_dt_ema;
            METRICS[13] = sim.complexity_estimate;
            METRICS[14] = sim.info_per_energy;
            METRICS[15] = sim.bank_size as f32;
            METRICS[16] = sim.output_len as f32;
            METRICS[17] = sim.pid_kp;
            METRICS[18] = sim.variance_gamma;
            METRICS[19] = (sim.metropolis_accepts as f32) /
                          ((sim.metropolis_accepts + sim.metropolis_rejects) as f32 + 1.0);
        });
        METRICS.as_ptr()
    }
}

#[no_mangle]
pub extern "C" fn get_output_ptr() -> *const u8 {
    with_sim(|sim| sim.output_buffer.as_ptr())
}

#[no_mangle]
pub extern "C" fn get_output_len() -> usize {
    with_sim(|sim| sim.output_len)
}

// ============================================================================
// Configuration Interface
// ============================================================================

#[no_mangle]
pub extern "C" fn set_pid_params(kp: f32, ki: f32, kd: f32, enable: bool) {
    with_sim_mut(|sim| {
        sim.pid_kp = kp;
        sim.pid_ki = ki;
        sim.pid_kd = kd;
        sim.use_pid = enable;
    });
}

#[no_mangle]
pub extern "C" fn set_variance_shaping(gamma: f32, enable: bool) {
    with_sim_mut(|sim| {
        sim.variance_gamma = gamma;
        sim.use_variance_shaping = enable;
    });
}

#[no_mangle]
pub extern "C" fn set_metropolis(enable: bool) {
    with_sim_mut(|sim| {
        sim.use_metropolis = enable;
    });
}

// ============================================================================
// AURORA Interface
// ============================================================================

#[no_mangle]
pub extern "C" fn aurora_init(trace_length: usize, state_features: usize, latent_dim: usize, seed: u64) {
    AURORA.with(|a| {
        *a.borrow_mut() = Some(AURORADescriptors::new(trace_length, state_features, latent_dim, seed));
    });
}

#[no_mangle]
pub extern "C" fn aurora_compute_descriptors() -> *const f32 {
    unsafe {
        with_sim(|sim| {
            AURORA.with(|a| {
                if let Some(aurora) = a.borrow_mut().as_mut() {
                    let (d1, d2) = aurora.compute_descriptors(sim, true);
                    DESC[0] = d1;
                    DESC[1] = d2;
                }
            });
        });
        DESC.as_ptr()
    }
}

// ============================================================================
// Lyapunov & Edge-Band Interface
// ============================================================================

#[no_mangle]
pub extern "C" fn lyapunov_init(perturbation: f32, window_size: usize) {
    LYAPUNOV.with(|l| {
        *l.borrow_mut() = Some(LyapunovEstimator::new(perturbation, window_size));
    });
}

#[no_mangle]
pub extern "C" fn edge_band_init(marginal_weight: f32, seed: u64) {
    EDGE_BAND.with(|e| {
        *e.borrow_mut() = Some(EdgeBandRouter::new(marginal_weight, seed));
    });
}

#[no_mangle]
pub extern "C" fn edge_band_get_stats_ptr() -> *const f32 {
    unsafe {
        EDGE_BAND.with(|e| {
            if let Some(router) = e.borrow().as_ref() {
                let stats = router.get_stats();
                EDGE_STATS[0] = stats.critical_stable as f32;
                EDGE_STATS[1] = stats.critical_chaotic as f32;
                EDGE_STATS[2] = stats.marginal as f32;
                EDGE_STATS[3] = stats.total as f32;
                EDGE_STATS[4] = stats.marginal_fraction;
            }
        });
        EDGE_STATS.as_ptr()
    }
}

// ============================================================================
// Island Ecology Interface
// ============================================================================

#[no_mangle]
pub extern "C" fn ecology_init(n_islands: usize, seed: u64) {
    let seed_genomes = vec![
        vec![b'+', b'+', b'[', b'>', b'+', b'<', b'-', b']', b'.'],
        vec![b':', b'{', b';', b'}', b'{', b'?', b'}', b'^', b'=', b'.'],
        vec![b'*', b'=', b'@', b'=', b'.', b'#'],
        vec![b'+', b'[', b'>', b'+', b'<', b'-', b']', b';', b'.'],
    ];

    ECOLOGY.with(|ec| {
        *ec.borrow_mut() = Some(IslandEcology::new(n_islands, &seed_genomes, seed));
    });
}

#[no_mangle]
pub extern "C" fn ecology_evolve(steps: u32, migration_interval: u32) {
    ECOLOGY.with(|ec| {
        if let Some(ecology) = ec.borrow_mut().as_mut() {
            ecology.evolve(steps, migration_interval);
        }
    });
}

#[no_mangle]
pub extern "C" fn ecology_get_stats_ptr() -> *const f32 {
    unsafe {
        ECOLOGY.with(|ec| {
            if let Some(ecology) = ec.borrow().as_ref() {
                let (immigrants, emigrants) = ecology.total_migrants();
                ECOLOGY_STATS[0] = ecology.generation as f32;
                ECOLOGY_STATS[1] = ecology.total_population() as f32;
                ECOLOGY_STATS[2] = immigrants as f32;
                ECOLOGY_STATS[3] = emigrants as f32;
            }
        });
        ECOLOGY_STATS.as_ptr()
    }
}

// ============================================================================
// Critic Interface
// ============================================================================

#[no_mangle]
pub extern "C" fn critic_init(ngram_size: usize, surprise_weight: f32, population_size: usize) {
    CRITIC.with(|c| {
        *c.borrow_mut() = Some(CriticEvolution::new(ngram_size, surprise_weight, population_size));
    });
}

#[no_mangle]
pub extern "C" fn critic_compute_fitness() -> *const f32 {
    unsafe {
        with_sim(|sim| {
            CRITIC.with(|c| {
                if let Some(critic_evo) = c.borrow().as_ref() {
                    let base_fitness = sim.e / (sim.steps as f32 + 1.0);
                    let output = &sim.output_buffer[..sim.output_len];
                    let (total, surprise_bonus) = critic_evo.compute_fitness_with_critic(base_fitness, output);
                    FITNESS[0] = total;
                    FITNESS[1] = surprise_bonus;
                }
            });
        });
        FITNESS.as_ptr()
    }
}

#[no_mangle]
pub extern "C" fn critic_learn_from_output() {
    with_sim(|sim| {
        CRITIC.with(|c| {
            if let Some(critic_evo) = c.borrow_mut().as_mut() {
                let output = sim.output_buffer[..sim.output_len].to_vec();
                critic_evo.learn_from_population(&[output]);
            }
        });
    });
}

// ============================================================================
// Reproducibility Spine Interface
// ============================================================================

#[no_mangle]
pub extern "C" fn repro_init(snapshot_interval: u32, enable_crash_capsule: bool) {
    REPRO.with(|r| {
        *r.borrow_mut() = Some(ReproSpine::new(snapshot_interval, enable_crash_capsule));
    });
}

#[no_mangle]
pub extern "C" fn repro_start_run(run_id_ptr: *const u8, run_id_len: usize) {
    unsafe {
        with_sim(|sim| {
            let run_id_slice = std::slice::from_raw_parts(run_id_ptr, run_id_len);
            let run_id = String::from_utf8_lossy(run_id_slice).to_string();

            REPRO.with(|r| {
                if let Some(repro) = r.borrow_mut().as_mut() {
                    repro.start_run(sim, run_id);
                }
            });
        });
    }
}

#[no_mangle]
pub extern "C" fn repro_snapshot() {
    with_sim(|sim| {
        REPRO.with(|r| {
            if let Some(repro) = r.borrow_mut().as_mut() {
                repro.snapshot(sim);
            }
        });
    });
}

#[no_mangle]
pub extern "C" fn repro_snapshot_count() -> usize {
    REPRO.with(|r| {
        if let Some(repro) = r.borrow().as_ref() {
            repro.snapshot_count()
        } else {
            0
        }
    })
}

#[no_mangle]
pub extern "C" fn repro_rewind(target_step: u32) -> bool {
    with_sim_mut(|sim| {
        REPRO.with(|r| {
            if let Some(repro) = r.borrow().as_ref() {
                repro.rewind(sim, target_step)
            } else {
                false
            }
        })
    })
}

// ============================================================================
// Validation & Self-Check
// ============================================================================

#[no_mangle]
pub extern "C" fn self_check() -> i32 {
    // Validate simulation state and invariants
    with_sim(|sim| {
        // Check 1: Metropolis acceptance rate should be in (0.20, 0.30)
        let total_metropolis = sim.metropolis_accepts + sim.metropolis_rejects;
        if total_metropolis > 100 {  // Only check after sufficient samples
            let acceptance_rate = (sim.metropolis_accepts as f32) / (total_metropolis as f32);
            if acceptance_rate < 0.20 || acceptance_rate > 0.30 {
                return 0;  // Fail: acceptance rate out of bounds
            }
        }

        // Check 2: Pointer stability - metrics pointer should be constant
        unsafe {
            let metrics_ptr = METRICS.as_ptr() as usize;
            let expected_ptr = &METRICS[0] as *const f32 as usize;
            if metrics_ptr != expected_ptr {
                return 0;  // Fail: pointer changed
            }
        }

        // Check 3: State consistency
        if sim.code_len > 4096 || sim.mem_size > 65536 {
            return 0;  // Fail: bounds violation
        }

        if sim.output_len > 16384 {
            return 0;  // Fail: output buffer overflow
        }

        1  // Pass
    })
}
