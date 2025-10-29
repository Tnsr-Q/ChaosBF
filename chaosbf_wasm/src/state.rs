#![allow(dead_code)]

use crate::rng::Rng;
use crate::ops::{Op, op_from_byte, delta_e, OPS};
use crate::thermo::local_entropy;
use std::collections::VecDeque;

const MAX_MEM: usize = 65536;
const MAX_CODE: usize = 4096;
const MAX_STACK: usize = 256;
const MAX_BANK: usize = 100;
const MAX_ELITE: usize = 50;
const MAX_BRANCH_HIST: usize = 200;
const MAX_OUTPUT: usize = 16384;

/// ChaosBF v4.0 simulation state with all advanced features
pub struct SimState {
    // Memory
    pub mem: [u8; MAX_MEM],
    pub mem_size: usize,
    pub ptr: usize,

    // Program
    pub code: [u8; MAX_CODE],
    pub code_len: usize,
    pub ip: usize,

    // Loop stack
    stack: [usize; MAX_STACK],
    stack_ptr: usize,

    // Thermodynamics
    pub e: f32,            // Energy
    pub t: f32,            // Temperature
    pub s: f32,            // Entropy accumulator
    pub f: f32,            // Free energy (E - T*S)

    // Evolution
    genome_bank: [[u8; MAX_CODE]; MAX_BANK],
    genome_lens: [usize; MAX_BANK],
    pub bank_size: usize,

    elite: [[u8; MAX_CODE]; MAX_ELITE],
    elite_lens: [usize; MAX_ELITE],
    elite_size: usize,

    // Criticality tracking
    pub lambda_hat: f32,
    branch_hist: [u32; MAX_BRANCH_HIST],
    branch_hist_ptr: usize,
    lambda_history: VecDeque<f32>,
    pub lambda_volatility: f32,

    pub steps: u32,

    // Settings
    pub tau: f32,              // Temperature adjustment
    pub theta_rep: f32,        // Replication threshold
    landauer_win: usize,

    // Local entropy cache
    pub slocal: f32,

    // PID controller state
    pub use_pid: bool,
    pub pid_kp: f32,
    pub pid_ki: f32,
    pub pid_kd: f32,
    pid_interval: u32,
    t_min: f32,
    t_max: f32,
    pid_integral: f32,
    pid_prev_error: f32,
    pid_updates: u32,

    // Variance shaping (dual-loop control)
    pub use_variance_shaping: bool,
    pub variance_gamma: f32,
    var_target: f32,
    var_alpha: f32,
    var_ema: f32,

    // Metropolis acceptance
    pub use_metropolis: bool,
    pub metropolis_accepts: u32,
    pub metropolis_rejects: u32,
    mutation_radius: f32,

    // Learning parameters
    learning_cap: f32,
    learning_rate_limit: f32,
    learning_energy_gained: f32,
    learning_window_start: u32,

    // Mutation parameters
    grammar_aware: bool,
    wild_mutation_rate: f32,
    adaptive_mutation: bool,

    // Leak parameters
    leak_max: f32,

    // Temporal derivatives (EMA)
    prev_s: f32,
    prev_k: f32,
    pub ds_dt_ema: f32,
    pub dk_dt_ema: f32,
    entropy_slope_alpha: f32,

    // Stats
    pub mutations: u32,
    mutations_wild: u32,
    pub replications: u32,
    pub crossovers: u32,
    pub learns: u32,

    // Output buffer
    pub output_buffer: [u8; MAX_OUTPUT],
    pub output_len: usize,

    // Complexity tracking
    pub complexity_estimate: f32,
    pub info_per_energy: f32,

    // RNG
    pub rng: Rng,
    pub seed: u64,
}

impl SimState {
    pub fn new(seed: u64, width: usize, height: usize, code: Vec<u8>) -> Self {
        let rng = Rng::from_seed(seed);
        let cells = width * height;
        let code_len = code.len().min(MAX_CODE);

        let mut state = Self {
            mem: [0; MAX_MEM],
            mem_size: cells.min(MAX_MEM),
            ptr: 0,
            code: [0; MAX_CODE],
            code_len,
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
            lambda_history: VecDeque::with_capacity(200),
            lambda_volatility: 0.0,
            steps: 0,
            tau: 0.1,
            theta_rep: 6.0,
            landauer_win: 16,
            slocal: 0.0,
            use_pid: true,
            pid_kp: 0.1,
            pid_ki: 0.01,
            pid_kd: 0.05,
            pid_interval: 50,
            t_min: 0.01,
            t_max: 2.0,
            pid_integral: 0.0,
            pid_prev_error: 0.0,
            pid_updates: 0,
            use_variance_shaping: true,
            variance_gamma: 0.02,
            var_target: 0.05,
            var_alpha: 0.01,
            var_ema: 0.0,
            use_metropolis: false,
            metropolis_accepts: 0,
            metropolis_rejects: 0,
            mutation_radius: 0.1,
            learning_cap: 0.5,
            learning_rate_limit: 0.05,
            learning_energy_gained: 0.0,
            learning_window_start: 0,
            grammar_aware: true,
            wild_mutation_rate: 0.1,
            adaptive_mutation: true,
            leak_max: 10.0,
            prev_s: 0.0,
            prev_k: 0.0,
            ds_dt_ema: 0.0,
            dk_dt_ema: 0.0,
            entropy_slope_alpha: 0.2,
            mutations: 0,
            mutations_wild: 0,
            replications: 0,
            crossovers: 0,
            learns: 0,
            output_buffer: [0; MAX_OUTPUT],
            output_len: 0,
            complexity_estimate: 0.0,
            info_per_energy: 0.0,
            rng,
            seed,
        };

        // Copy code
        state.code[..code_len].copy_from_slice(&code[..code_len]);

        state
    }

    #[inline]
    pub fn branching_factor(&self) -> f32 {
        if self.branch_hist_ptr == 0 {
            return 1.0;
        }
        let sum: u32 = self.branch_hist[..self.branch_hist_ptr].iter().sum();
        sum as f32 / self.branch_hist_ptr as f32
    }

    #[inline]
    pub fn sense_entropy(&self) -> f32 {
        let w = 8;
        let a = self.ptr.saturating_sub(w);
        let b = (self.ptr + w + 1).min(self.mem_size);
        local_entropy(&self.mem[a..b])
    }

    #[inline]
    pub fn update_free_energy(&mut self) {
        self.f = self.e - self.t * self.s;
    }

    #[inline]
    fn update_lambda_volatility(&mut self) {
        if self.lambda_history.len() >= 2 {
            // Compute rolling std dev
            let mean: f32 = self.lambda_history.iter().sum::<f32>() / self.lambda_history.len() as f32;
            let variance: f32 = self.lambda_history.iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f32>() / self.lambda_history.len() as f32;
            self.lambda_volatility = variance.sqrt();
        }
    }

    /// PID controller for criticality maintenance
    fn pid_update(&mut self) {
        if !self.use_pid {
            return;
        }

        let lambda_target = 1.0;
        let error = self.lambda_hat - lambda_target;

        // PID terms
        self.pid_integral += error;
        let derivative = error - self.pid_prev_error;

        let correction = self.pid_kp * error
            + self.pid_ki * self.pid_integral
            + self.pid_kd * derivative;

        // Apply correction to temperature
        self.t = (self.t - correction).clamp(self.t_min, self.t_max);

        self.pid_prev_error = error;
        self.pid_updates += 1;
    }

    /// Variance shaping (dual-loop control)
    fn variance_shaping_update(&mut self) {
        if !self.use_variance_shaping {
            return;
        }

        // Update variance EMA
        if self.lambda_history.len() >= 2 {
            let mean: f32 = self.lambda_history.iter().sum::<f32>() / self.lambda_history.len() as f32;
            let variance: f32 = self.lambda_history.iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f32>() / self.lambda_history.len() as f32;

            self.var_ema = self.var_alpha * variance + (1.0 - self.var_alpha) * self.var_ema;

            // Adjust temperature based on variance deficit
            let var_error = self.var_ema - self.var_target;
            let var_correction = self.variance_gamma * var_error;

            self.t = (self.t + var_correction).clamp(self.t_min, self.t_max);
        }
    }

    /// Update temporal derivatives
    fn update_derivatives(&mut self) {
        if self.steps > 0 {
            let ds = self.s - self.prev_s;
            let dk = self.complexity_estimate - self.prev_k;

            // EMA smoothing
            self.ds_dt_ema = self.entropy_slope_alpha * ds
                + (1.0 - self.entropy_slope_alpha) * self.ds_dt_ema;
            self.dk_dt_ema = self.entropy_slope_alpha * dk
                + (1.0 - self.entropy_slope_alpha) * self.dk_dt_ema;
        }

        self.prev_s = self.s;
        self.prev_k = self.complexity_estimate;
    }

    /// Metropolis acceptance criterion
    fn metropolis_accept(&mut self, delta_f: f32) -> bool {
        if !self.use_metropolis {
            return true;
        }

        if delta_f <= 0.0 {
            // Always accept improvements
            self.metropolis_accepts += 1;
            true
        } else {
            // Accept with probability exp(-ΔF / T)
            let prob = (-delta_f / self.t).exp();
            let accept = self.rng.gen_f32() < prob;

            if accept {
                self.metropolis_accepts += 1;
            } else {
                self.metropolis_rejects += 1;
            }

            accept
        }
    }

    /// Execute a single step
    pub fn step(&mut self) {
        if self.e <= 0.0 || self.code_len == 0 {
            return;
        }

        self.ip = self.ip % self.code_len;
        let op_byte = self.code[self.ip];
        let op = op_from_byte(op_byte);

        // Sense entropy for ':' operator
        if op == Op::Colon {
            self.slocal = self.sense_entropy();
        }

        // Energy cost
        let de = delta_e(op, self.stack_ptr, self.slocal);
        self.e += de;

        // Execute operator
        self.execute_op(op);

        self.ip += 1;
        self.steps += 1;

        // Update free energy
        self.update_free_energy();

        // Update lambda if we have history
        if self.steps % 10 == 0 {
            let lambda = self.branching_factor();
            self.lambda_hat = lambda;
            self.lambda_history.push_back(lambda);
            if self.lambda_history.len() > 200 {
                self.lambda_history.pop_front();
            }
            self.update_lambda_volatility();
        }

        // PID control
        if self.use_pid && self.steps % self.pid_interval == 0 {
            self.pid_update();
        }

        // Variance shaping
        if self.use_variance_shaping && self.steps % self.pid_interval == 0 {
            self.variance_shaping_update();
        }

        // Update derivatives
        if self.steps % 100 == 0 {
            self.update_derivatives();
        }
    }

    fn execute_op(&mut self, op: Op) {
        let mem_size = self.mem_size;

        match op {
            Op::Gt => self.ptr = (self.ptr + 1) % mem_size,
            Op::Lt => self.ptr = (self.ptr + mem_size - 1) % mem_size,
            Op::Plus => self.mem[self.ptr] = self.mem[self.ptr].wrapping_add(1),
            Op::Minus => self.mem[self.ptr] = self.mem[self.ptr].wrapping_sub(1),
            Op::Dot => {
                if self.output_len < MAX_OUTPUT {
                    self.output_buffer[self.output_len] = self.mem[self.ptr];
                    self.output_len += 1;
                }
            },
            Op::Comma => self.mem[self.ptr] = 0,

            Op::LBr => {
                if self.mem[self.ptr] == 0 {
                    let mut bal = 1usize;
                    while bal > 0 && self.ip + 1 < self.code_len {
                        self.ip += 1;
                        match self.code[self.ip] {
                            b'[' => bal += 1,
                            b']' => bal -= 1,
                            _ => {}
                        }
                    }
                } else if self.stack_ptr < MAX_STACK {
                    self.stack[self.stack_ptr] = self.ip;
                    self.stack_ptr += 1;
                }
            }

            Op::RBr => {
                if self.mem[self.ptr] != 0 && self.stack_ptr > 0 {
                    self.ip = self.stack[self.stack_ptr - 1];
                } else if self.stack_ptr > 0 {
                    self.stack_ptr -= 1;
                }
            }

            Op::Caret => self.t = (self.t + self.tau).min(2.0),
            Op::Vee => self.t = (self.t - self.tau).max(0.01),
            Op::Colon => {},
            Op::Semi => self.s += self.slocal,

            Op::Q => self.mutate_op(),
            Op::Star => self.replicate_op(),
            Op::At => self.crossover_op(),
            Op::Eq => self.learn_op(),
            Op::Bang => self.elite_save_op(),
            Op::Tilde => self.elite_load_op(),

            Op::LCurly => self.branch_op(),
            Op::RCurly => {},

            Op::Hash => self.complexity_op(),
            Op::Percent => self.info_per_energy_op(),

            Op::Unknown => {}
        }
    }

    fn mutate_op(&mut self) {
        let p_mut = (0.2 + 0.6 * self.t).min(0.95);
        if self.rng.gen_f32() < p_mut && self.code_len > 0 {
            let i = self.rng.gen_range(0, self.code_len);
            self.code[i] = OPS[self.rng.gen_range(0, OPS.len())];
            self.mutations += 1;
        }
    }

    fn replicate_op(&mut self) {
        if self.f > self.theta_rep && self.bank_size < MAX_BANK {
            self.genome_bank[self.bank_size][..self.code_len].copy_from_slice(&self.code[..self.code_len]);
            self.genome_lens[self.bank_size] = self.code_len;
            self.bank_size += 1;

            let burst = ((self.t * 3.0).max(1.0) as usize).min(self.code_len);
            for _ in 0..burst {
                if self.code_len > 0 {
                    let i = self.rng.gen_range(0, self.code_len);
                    self.code[i] = OPS[self.rng.gen_range(0, OPS.len())];
                }
            }

            self.replications += 1;
        }
    }

    fn crossover_op(&mut self) {
        if self.bank_size > 0 && self.code_len > 2 {
            let idx = self.rng.gen_range(0, self.bank_size);
            let mate_len = self.genome_lens[idx];
            let k = self.code_len.min(mate_len);

            if k > 2 {
                let cut = self.rng.gen_range(1, k);
                self.code[cut..k].copy_from_slice(&self.genome_bank[idx][cut..k]);
            }

            self.crossovers += 1;
        }
    }

    fn learn_op(&mut self) {
        let mut opt = [0u8; MAX_CODE];
        let mut opt_len = 0;
        let mut i = 0;

        while i < self.code_len {
            if i + 1 < self.code_len {
                let pair = (self.code[i], self.code[i + 1]);
                match pair {
                    (b'+', b'-') | (b'-', b'+') | (b'>', b'<') | (b'<', b'>') => {
                        i += 2;
                        continue;
                    }
                    _ => {}
                }
            }
            if opt_len < MAX_CODE {
                opt[opt_len] = self.code[i];
                opt_len += 1;
            }
            i += 1;
        }

        let saved = self.code_len.saturating_sub(opt_len);
        if saved > 0 {
            self.code[..opt_len].copy_from_slice(&opt[..opt_len]);
            self.code_len = opt_len;
            self.e += (saved as f32) * 0.5;
            self.learns += 1;
        }
    }

    fn elite_save_op(&mut self) {
        if self.e > 50.0 && self.s > 1.0 && self.elite_size < MAX_ELITE {
            self.elite[self.elite_size][..self.code_len].copy_from_slice(&self.code[..self.code_len]);
            self.elite_lens[self.elite_size] = self.code_len;
            self.elite_size += 1;
        }
    }

    fn elite_load_op(&mut self) {
        if self.f < 0.0 && self.elite_size > 0 {
            let idx = self.rng.gen_range(0, self.elite_size);
            let elite_len = self.elite_lens[idx];
            self.code[..elite_len].copy_from_slice(&self.elite[idx][..elite_len]);
            self.code_len = elite_len;
            self.t *= 0.8;
            self.ip = 0;
            self.stack_ptr = 0;
        }
    }

    fn branch_op(&mut self) {
        let p_branch = (0.3 + 0.4 * self.t).min(0.9);
        if self.rng.gen_f32() < p_branch {
            self.mem[self.ptr] ^= 1;
            if self.branch_hist_ptr < MAX_BRANCH_HIST {
                self.branch_hist[self.branch_hist_ptr] = 1;
                self.branch_hist_ptr += 1;
            }
        } else if self.branch_hist_ptr < MAX_BRANCH_HIST {
            self.branch_hist[self.branch_hist_ptr] = 0;
            self.branch_hist_ptr += 1;
        }
    }

    fn complexity_op(&mut self) {
        // Approximate Kolmogorov complexity via output length
        if self.output_len > 0 {
            self.complexity_estimate = self.output_len as f32;
        }
    }

    fn info_per_energy_op(&mut self) {
        if self.e > 0.0 {
            self.info_per_energy = self.complexity_estimate / self.e;
        }
    }
}
