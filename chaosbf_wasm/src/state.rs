use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

pub struct SimState {
    // memory & pointer
    pub mem: Vec<u8>,
    pub ptr: usize,

    // program
    pub code: Vec<u8>,
    pub ip: usize,
    pub loop_stack: Vec<usize>,

    // thermodynamics
    pub e: f32,   // energy
    pub t: f32,   // temperature
    pub s: f32,   // entropy accumulator
    pub f: f32,   // free energy (derived)

    // control/metrics
    pub lambda_hat: f32,
    pub branch_hist: Vec<u32>,
    pub steps: u32,

    // evolution
    pub genome_bank: Vec<Vec<u8>>,
    pub elite: Vec<Vec<u8>>,

    // rng
    pub rng: ChaCha20Rng,

    // viz grid (reshape tape) dims
    pub w: usize,
    pub h: usize,

    // settings
    pub tau: f32,           // temp adjustment
    pub theta_rep: f32,     // replication threshold
    pub landauer_win: usize,

    // local entropy cache
    pub slocal: f32,

    // stats
    pub mutations: u32,
    pub replications: u32,
    pub crossovers: u32,
    pub learns: u32,
}

impl SimState {
    pub fn new(seed: u64, width: usize, height: usize, code: Vec<u8>) -> Self {
        let rng = ChaCha20Rng::seed_from_u64(seed);
        let cells = width * height;
        Self {
            mem: vec![0; cells],
            ptr: 0,
            code,
            ip: 0,
            loop_stack: Vec::with_capacity(64),
            e: 200.0,
            t: 0.6,
            s: 0.0,
            f: 0.0,
            lambda_hat: 1.0,
            branch_hist: Vec::with_capacity(200),
            steps: 0,
            genome_bank: Vec::new(),
            elite: Vec::new(),
            rng,
            w: width,
            h: height,
            tau: 0.1,
            theta_rep: 6.0,
            landauer_win: 16,
            slocal: 0.0,
            mutations: 0,
            replications: 0,
            crossovers: 0,
            learns: 0,
        }
    }

    #[inline]
    pub fn shape_to_frame(&self) -> &[u8] {
        &self.mem
    }

    #[inline]
    pub fn update_free_energy(&mut self) {
        self.f = self.e - self.t * self.s;
    }

    #[inline]
    pub fn branching_factor(&self) -> f32 {
        if self.branch_hist.is_empty() {
            return 1.0;
        }
        let sum: u32 = self.branch_hist.iter().sum();
        sum as f32 / self.branch_hist.len() as f32
    }
}
