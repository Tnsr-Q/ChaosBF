//! ChaosBF WASM - Standalone version without external dependencies
//! Uses raw WASM exports and minimal stdlib

#![no_std]

use core::panic::PanicInfo;

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}

// Simple LCG PRNG (no external deps)
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self.state.wrapping_mul(6364136223846793005)
                              .wrapping_add(1442695040888963407);
        (self.state >> 32) as u32
    }

    fn gen_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / 16777216.0
    }

    fn gen_range(&mut self, min: usize, max: usize) -> usize {
        if max <= min {
            return min;
        }
        min + (self.next_u32() as usize) % (max - min)
    }
}

// Global simulation state (single instance)
static mut SIM: Option<SimState> = None;

const MAX_MEM: usize = 65536;
const MAX_CODE: usize = 4096;
const MAX_STACK: usize = 256;
const MAX_BANK: usize = 100;
const MAX_ELITE: usize = 50;
const MAX_BRANCH_HIST: usize = 200;

struct SimState {
    // Memory
    mem: [u8; MAX_MEM],
    mem_size: usize,
    ptr: usize,

    // Program
    code: [u8; MAX_CODE],
    code_len: usize,
    ip: usize,

    // Loop stack
    stack: [usize; MAX_STACK],
    stack_ptr: usize,

    // Thermodynamics
    e: f32,
    t: f32,
    s: f32,
    f: f32,

    // Evolution
    genome_bank: [[u8; MAX_CODE]; MAX_BANK],
    genome_lens: [usize; MAX_BANK],
    bank_size: usize,

    elite: [[u8; MAX_CODE]; MAX_ELITE],
    elite_lens: [usize; MAX_ELITE],
    elite_size: usize,

    // Metrics
    lambda_hat: f32,
    branch_hist: [u32; MAX_BRANCH_HIST],
    branch_hist_ptr: usize,
    steps: u32,

    // Settings
    tau: f32,
    theta_rep: f32,
    landauer_win: usize,
    slocal: f32,

    // Stats
    mutations: u32,
    replications: u32,
    crossovers: u32,
    learns: u32,

    // RNG
    rng: Rng,
}

impl SimState {
    fn new() -> Self {
        Self {
            mem: [0; MAX_MEM],
            mem_size: 0,
            ptr: 0,
            code: [0; MAX_CODE],
            code_len: 0,
            ip: 0,
            stack: [0; MAX_STACK],
            stack_ptr: 0,
            e: 200.0,
            t: 0.6,
            s: 0.0,
            f: 200.0,
            genome_bank: [[0; MAX_CODE]; MAX_BANK],
            genome_lens: [0; MAX_BANK],
            bank_size: 0,
            elite: [[0; MAX_CODE]; MAX_ELITE],
            elite_lens: [0; MAX_ELITE],
            elite_size: 0,
            lambda_hat: 1.0,
            branch_hist: [0; MAX_BRANCH_HIST],
            branch_hist_ptr: 0,
            steps: 0,
            tau: 0.1,
            theta_rep: 6.0,
            landauer_win: 16,
            slocal: 0.0,
            mutations: 0,
            replications: 0,
            crossovers: 0,
            learns: 0,
            rng: Rng::new(12345),
        }
    }

    fn branching_factor(&self) -> f32 {
        if self.branch_hist_ptr == 0 {
            return 1.0;
        }
        let sum: u32 = self.branch_hist[..self.branch_hist_ptr].iter().sum();
        sum as f32 / self.branch_hist_ptr as f32
    }

    fn sense_entropy(&self) -> f32 {
        let w = 8;
        let a = self.ptr.saturating_sub(w);
        let b = (self.ptr + w + 1).min(self.mem_size);
        local_entropy(&self.mem[a..b])
    }
}

fn local_entropy(bytes: &[u8]) -> f32 {
    if bytes.is_empty() {
        return 0.0;
    }

    let mut hist = [0u16; 256];
    for &b in bytes {
        hist[b as usize] = hist[b as usize].saturating_add(1);
    }

    let n = bytes.len() as f32;
    let mut h = 0.0;

    for &count in hist.iter() {
        if count == 0 { continue; }
        let p = (count as f32) / n;
        // Simple log approximation
        h -= p * fast_ln(p);
    }

    h
}

// Fast ln approximation
fn fast_ln(x: f32) -> f32 {
    if x <= 0.0 { return -10.0; }
    let mut y = x;
    let mut log = 0.0;

    while y > 2.0 {
        y /= 2.0;
        log += 0.69314718; // ln(2)
    }

    let z = y - 1.0;
    z - 0.5 * z * z + 0.333 * z * z * z
    + log
}

fn delta_e(op: u8, depth: usize, slocal: f32) -> f32 {
    let leak = if matches!(op, b'[' | b']' | b'{' | b'}') {
        1.0 + (depth as f32) / 3.0
    } else {
        0.0
    };

    let base = match op {
        b'>' | b'<' => -1.0,
        b'+' => -2.0,
        b'-' => 1.0,
        b'.' | b',' => -1.0,
        b'^' | b'v' => -1.0,
        b':' => 0.0,
        b';' => -slocal,
        b'?' => -2.0,
        b'*' => -10.0,
        b'@' => -6.0,
        b'=' => 0.0,
        b'!' => -1.0,
        b'{' => -2.0,
        b'}' => 0.0,
        b'#' => 0.0,
        b'%' => -1.0,
        b'~' => 5.0,
        _ => 0.0,
    };
    base - leak
}

const OPS: &[u8] = b"><+-[].,^v:;?*@=!{}#%~";

fn execute_op(sim: &mut SimState, op: u8) {
    let mem_size = sim.mem_size;

    match op {
        b'>' => sim.ptr = (sim.ptr + 1) % mem_size,
        b'<' => sim.ptr = (sim.ptr + mem_size - 1) % mem_size,
        b'+' => sim.mem[sim.ptr] = sim.mem[sim.ptr].wrapping_add(1),
        b'-' => sim.mem[sim.ptr] = sim.mem[sim.ptr].wrapping_sub(1),
        b'.' => {},
        b',' => sim.mem[sim.ptr] = 0,

        b'[' => {
            if sim.mem[sim.ptr] == 0 {
                let mut bal = 1usize;
                while bal > 0 && sim.ip + 1 < sim.code_len {
                    sim.ip += 1;
                    match sim.code[sim.ip] {
                        b'[' => bal += 1,
                        b']' => bal -= 1,
                        _ => {}
                    }
                }
            } else if sim.stack_ptr < MAX_STACK {
                sim.stack[sim.stack_ptr] = sim.ip;
                sim.stack_ptr += 1;
            }
        }

        b']' => {
            if sim.mem[sim.ptr] != 0 && sim.stack_ptr > 0 {
                sim.ip = sim.stack[sim.stack_ptr - 1];
            } else if sim.stack_ptr > 0 {
                sim.stack_ptr -= 1;
            }
        }

        b'^' => sim.t = (sim.t + sim.tau).min(2.0),
        b'v' => sim.t = (sim.t - sim.tau).max(0.01),
        b':' => {},
        b';' => sim.s += sim.slocal,

        b'?' => {
            let p_mut = (0.2 + 0.6 * sim.t).min(0.95);
            if sim.rng.gen_f32() < p_mut && sim.code_len > 0 {
                let i = sim.rng.gen_range(0, sim.code_len);
                sim.code[i] = OPS[sim.rng.gen_range(0, OPS.len())];
                sim.mutations += 1;
            }
        }

        b'*' => {
            if sim.f > sim.theta_rep && sim.bank_size < MAX_BANK {
                // Add to bank
                sim.genome_bank[sim.bank_size][..sim.code_len].copy_from_slice(&sim.code[..sim.code_len]);
                sim.genome_lens[sim.bank_size] = sim.code_len;
                sim.bank_size += 1;

                // Mutation burst
                let burst = ((sim.t * 3.0).max(1.0) as usize).min(sim.code_len);
                for _ in 0..burst {
                    if sim.code_len > 0 {
                        let i = sim.rng.gen_range(0, sim.code_len);
                        sim.code[i] = OPS[sim.rng.gen_range(0, OPS.len())];
                    }
                }

                sim.replications += 1;
            }
        }

        b'@' => {
            if sim.bank_size > 0 && sim.code_len > 2 {
                let idx = sim.rng.gen_range(0, sim.bank_size);
                let mate_len = sim.genome_lens[idx];
                let k = sim.code_len.min(mate_len);

                if k > 2 {
                    let cut = sim.rng.gen_range(1, k);
                    // Crossover
                    sim.code[cut..k].copy_from_slice(&sim.genome_bank[idx][cut..k]);
                }

                sim.crossovers += 1;
            }
        }

        b'=' => {
            // Simple peephole optimization
            let mut opt = [0u8; MAX_CODE];
            let mut opt_len = 0;
            let mut i = 0;

            while i < sim.code_len {
                if i + 1 < sim.code_len {
                    let pair = (sim.code[i], sim.code[i + 1]);
                    match pair {
                        (b'+', b'-') | (b'-', b'+') | (b'>', b'<') | (b'<', b'>') => {
                            i += 2;
                            continue;
                        }
                        _ => {}
                    }
                }
                if opt_len < MAX_CODE {
                    opt[opt_len] = sim.code[i];
                    opt_len += 1;
                }
                i += 1;
            }

            let saved = sim.code_len.saturating_sub(opt_len);
            if saved > 0 {
                sim.code[..opt_len].copy_from_slice(&opt[..opt_len]);
                sim.code_len = opt_len;
                sim.e += (saved as f32) * 0.5;
                sim.learns += 1;
            }
        }

        b'!' => {
            if sim.e > 50.0 && sim.s > 1.0 && sim.elite_size < MAX_ELITE {
                sim.elite[sim.elite_size][..sim.code_len].copy_from_slice(&sim.code[..sim.code_len]);
                sim.elite_lens[sim.elite_size] = sim.code_len;
                sim.elite_size += 1;
            }
        }

        b'{' => {
            let p_branch = (0.3 + 0.4 * sim.t).min(0.9);
            if sim.rng.gen_f32() < p_branch {
                sim.mem[sim.ptr] ^= 1;
                if sim.branch_hist_ptr < MAX_BRANCH_HIST {
                    sim.branch_hist[sim.branch_hist_ptr] = 1;
                    sim.branch_hist_ptr += 1;
                }
            } else if sim.branch_hist_ptr < MAX_BRANCH_HIST {
                sim.branch_hist[sim.branch_hist_ptr] = 0;
                sim.branch_hist_ptr += 1;
            }
        }

        b'~' => {
            if sim.f < 0.0 && sim.elite_size > 0 {
                let idx = sim.rng.gen_range(0, sim.elite_size);
                let elite_len = sim.elite_lens[idx];
                sim.code[..elite_len].copy_from_slice(&sim.elite[idx][..elite_len]);
                sim.code_len = elite_len;
                sim.t *= 0.8;
                sim.ip = 0;
                sim.stack_ptr = 0;
            }
        }

        _ => {}
    }
}

// WASM exports
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
    unsafe {
        let mut sim = SimState::new();
        sim.rng = Rng::new(seed);
        sim.mem_size = width * height;
        sim.e = e0;
        sim.t = t0;

        // Copy code
        let code_slice = core::slice::from_raw_parts(code_ptr, code_len.min(MAX_CODE));
        sim.code_len = code_slice.len();
        sim.code[..sim.code_len].copy_from_slice(code_slice);

        SIM = Some(sim);
    }
}

#[no_mangle]
pub extern "C" fn step_sim(ticks: u32) {
    unsafe {
        if let Some(sim) = &mut SIM {
            for _ in 0..ticks {
                if sim.e <= 0.0 || sim.code_len == 0 {
                    break;
                }

                sim.ip = sim.ip % sim.code_len;
                let op = sim.code[sim.ip];

                if op == b':' {
                    sim.slocal = sim.sense_entropy();
                }

                sim.e += delta_e(op, sim.stack_ptr, sim.slocal);

                execute_op(sim, op);

                sim.ip += 1;
                sim.steps += 1;
                sim.f = sim.e - sim.t * sim.s;
            }

            sim.lambda_hat = sim.branching_factor();
        }
    }
}

#[no_mangle]
pub extern "C" fn get_mem_ptr() -> *const u8 {
    unsafe {
        if let Some(sim) = &SIM {
            sim.mem.as_ptr()
        } else {
            core::ptr::null()
        }
    }
}

#[no_mangle]
pub extern "C" fn get_metrics_ptr() -> *const f32 {
    unsafe {
        static mut METRICS: [f32; 10] = [0.0; 10];

        if let Some(sim) = &SIM {
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
        }

        METRICS.as_ptr()
    }
}
