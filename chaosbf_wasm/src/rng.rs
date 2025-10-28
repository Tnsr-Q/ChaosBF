/// Self-contained pseudo-random number generator (PCG variant)
/// No external dependencies
#[derive(Clone, Copy, Debug)]
pub struct Rng {
    state: u64,
    inc: u64,
}

impl Rng {
    /// Create new RNG from seed
    pub fn from_seed(seed: u64) -> Self {
        let mut rng = Self {
            state: 0,
            inc: (seed << 1) | 1,
        };
        rng.state = rng.next_u64();
        rng.state = rng.state.wrapping_add(seed);
        rng.next_u64();
        rng
    }

    /// Generate next u64
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let oldstate = self.state;
        self.state = oldstate
            .wrapping_mul(6364136223846793005)
            .wrapping_add(self.inc);

        let xorshifted = ((oldstate >> 18) ^ oldstate) >> 27;
        let rot = (oldstate >> 59) as u32;

        ((xorshifted >> rot) | (xorshifted << ((!rot).wrapping_add(1) & 31))) as u64
    }

    /// Generate u32
    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    /// Generate f32 in [0, 1)
    #[inline]
    pub fn gen_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / 16777216.0
    }

    /// Generate f64 in [0, 1)
    #[inline]
    pub fn gen_f64(&mut self) -> f64 {
        let upper = (self.next_u64() >> 11) as f64;
        upper / 9007199254740992.0
    }

    /// Generate usize in range [min, max)
    #[inline]
    pub fn gen_range(&mut self, min: usize, max: usize) -> usize {
        if max <= min {
            return min;
        }
        let range = (max - min) as u32;
        let threshold = range.wrapping_neg() % range;

        loop {
            let r = self.next_u32();
            if r >= threshold {
                return min + ((r % range) as usize);
            }
        }
    }

    /// Generate bool with given probability
    #[inline]
    pub fn gen_bool(&mut self, p: f32) -> bool {
        self.gen_f32() < p
    }

    /// Fill slice with random bytes
    pub fn fill_bytes(&mut self, dest: &mut [u8]) {
        let mut left = dest.len();
        let mut i = 0;

        while left >= 8 {
            let val = self.next_u64();
            dest[i..i+8].copy_from_slice(&val.to_le_bytes());
            left -= 8;
            i += 8;
        }

        if left > 0 {
            let val = self.next_u64();
            dest[i..].copy_from_slice(&val.to_le_bytes()[..left]);
        }
    }

    /// Shuffle a slice
    pub fn shuffle<T>(&mut self, slice: &mut [T]) {
        let len = slice.len();
        for i in (1..len).rev() {
            let j = self.gen_range(0, i + 1);
            slice.swap(i, j);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rng_deterministic() {
        let mut rng1 = Rng::from_seed(42);
        let mut rng2 = Rng::from_seed(42);

        for _ in 0..100 {
            assert_eq!(rng1.next_u32(), rng2.next_u32());
        }
    }

    #[test]
    fn test_gen_range() {
        let mut rng = Rng::from_seed(12345);

        for _ in 0..100 {
            let val = rng.gen_range(10, 20);
            assert!(val >= 10 && val < 20);
        }
    }

    #[test]
    fn test_gen_f32() {
        let mut rng = Rng::from_seed(67890);

        for _ in 0..100 {
            let val = rng.gen_f32();
            assert!(val >= 0.0 && val < 1.0);
        }
    }
}
