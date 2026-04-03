/// Tau random number generator based on the Python PyNNDescent library (https://github.com/lmcinnes/pynndescent).
/// Uses a 3-element i64 state with specific bitmask operations.

#[derive(Clone, Debug)]
pub struct TauRng {
    pub state: [i64; 3],
}

impl TauRng {
    /// Create a new TauRng seeded with the given value.
    /// Matches Python: `rng_state.fill(seed + 0xFFFF)`
    pub fn new(seed: i64) -> Self {
        let val = seed.wrapping_add(0xFFFF);
        TauRng {
            state: [val, val, val],
        }
    }

    /// Create from an existing 3-element state.
    pub fn from_state(state: [i64; 3]) -> Self {
        TauRng { state }
    }

    /// A fast (pseudo)-random number generator.
    /// Must exactly match the Python bitmask arithmetic.
    ///
    /// The Python code uses i64 state but performs u32-level bit operations:
    /// - `& 0xFFFFFFFF` ensures 32-bit truncation
    /// - Constants: 4294967294 = 0xFFFFFFFE, 4294967288 = 0xFFFFFFF8, 4294967280 = 0xFFFFFFF0
    pub fn tau_rand_int(&mut self) -> i32 {
        // State 0: mask=0xFFFFFFFE, shift_left=12, shift_right=19, shift2=13
        self.state[0] = {
            let s = self.state[0];
            let a = ((s & 0xFFFFFFFE_i64) << 12) & 0xFFFFFFFF;
            let b = (((s << 13) & 0xFFFFFFFF) ^ s) >> 19;
            (a ^ b) as i64
        };

        // State 1: mask=0xFFFFFFF8, shift_left=4, shift_right=25, shift2=2
        self.state[1] = {
            let s = self.state[1];
            let a = ((s & 0xFFFFFFF8_i64) << 4) & 0xFFFFFFFF;
            let b = (((s << 2) & 0xFFFFFFFF) ^ s) >> 25;
            (a ^ b) as i64
        };

        // State 2: mask=0xFFFFFFF0, shift_left=17, shift_right=11, shift2=3
        self.state[2] = {
            let s = self.state[2];
            let a = ((s & 0xFFFFFFF0_i64) << 17) & 0xFFFFFFFF;
            let b = (((s << 3) & 0xFFFFFFFF) ^ s) >> 11;
            (a ^ b) as i64
        };

        (self.state[0] ^ self.state[1] ^ self.state[2]) as i32
    }

    /// A fast (pseudo)-random number generator for floats in [0, 1].
    pub fn tau_rand(&mut self) -> f32 {
        let integer = self.tau_rand_int();
        (integer as f64 / 0x7FFFFFFF_u32 as f64).abs() as f32
    }

    /// Generate n_samples many integers from 0 to pool_size such that no
    /// integer is selected twice. Uses rejection sampling.
    pub fn rejection_sample(&mut self, n_samples: usize, pool_size: usize) -> Vec<i64> {
        let mut result = Vec::with_capacity(n_samples);
        for i in 0..n_samples {
            loop {
                let j = (self.tau_rand_int() % pool_size as i32).abs() as i64;
                let mut duplicate = false;
                for k in 0..i {
                    if j == result[k] {
                        duplicate = true;
                        break;
                    }
                }
                if !duplicate {
                    result.push(j);
                    break;
                }
            }
        }
        result
    }
}

// ---------------------------------------------------------------------------
// Xoshiro256** — general-purpose PRNG
// Reference: https://prng.di.unimi.it/xoshiro256starstar.c
// ---------------------------------------------------------------------------

/// A fast, high-quality 256-bit PRNG (xoshiro256**).
#[derive(Clone, Debug)]
pub struct Xoshiro256StarStar {
    s: [u64; 4],
}

impl Xoshiro256StarStar {
    /// Seed from a single u64 using splitmix64 to fill the 256-bit state.
    pub fn seed_from_u64(seed: u64) -> Self {
        let mut z = seed;
        let mut s = [0u64; 4];
        for slot in &mut s {
            z = z.wrapping_add(0x9e3779b97f4a7c15);
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            *slot = z ^ (z >> 31);
        }
        Xoshiro256StarStar { s }
    }

    /// Seed from OS entropy via `getrandom`.
    pub fn seed_from_os() -> Self {
        let mut buf = [0u8; 32];
        getrandom::fill(&mut buf).expect("failed to get OS entropy");
        let s = [
            u64::from_le_bytes(buf[0..8].try_into().unwrap()),
            u64::from_le_bytes(buf[8..16].try_into().unwrap()),
            u64::from_le_bytes(buf[16..24].try_into().unwrap()),
            u64::from_le_bytes(buf[24..32].try_into().unwrap()),
        ];
        Xoshiro256StarStar { s }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let result = (self.s[1].wrapping_mul(5)).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform f64 in [0, 1).
    #[inline]
    pub fn random_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform f32 in [0, 1).
    #[inline]
    pub fn random_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 * (1.0 / (1u64 << 24) as f32)
    }

    /// Random i64 (full range).
    #[inline]
    pub fn random_i64(&mut self) -> i64 {
        self.next_u64() as i64
    }

    /// Random i64 in [lo, hi).
    pub fn random_range_i64(&mut self, lo: i64, hi: i64) -> i64 {
        debug_assert!(lo < hi);
        let range = (hi as i128 - lo as i128) as u64;
        lo.wrapping_add((self.next_u64() % range) as i64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tau_rng_deterministic() {
        let mut rng1 = TauRng::new(42);
        let mut rng2 = TauRng::new(42);
        for _ in 0..100 {
            assert_eq!(rng1.tau_rand_int(), rng2.tau_rand_int());
        }
    }

    #[test]
    fn test_tau_rand_range() {
        let mut rng = TauRng::new(42);
        for _ in 0..1000 {
            let val = rng.tau_rand();
            assert!(val >= 0.0 && val <= 1.0, "tau_rand out of range: {}", val);
        }
    }

    #[test]
    fn test_tau_rng_entropy() {
        // Shannon entropy of tau_rand_int output bucketed into 256 bins.
        // A perfect uniform source has entropy = ln(256) ≈ 5.545.
        let n = 100_000;
        let bins = 256usize;
        let mut counts = vec![0u32; bins];
        let mut rng = TauRng::new(42);
        for _ in 0..n {
            let val = rng.tau_rand_int();
            let bucket = (val as u32) % bins as u32;
            counts[bucket as usize] += 1;
        }
        let n_f = n as f64;
        let entropy: f64 = counts
            .iter()
            .filter(|&&c| c > 0)
            .map(|&c| {
                let p = c as f64 / n_f;
                -p * p.ln()
            })
            .sum();
        let max_entropy = (bins as f64).ln();
        // Require at least 99% of maximum entropy.
        assert!(
            entropy > max_entropy * 0.99,
            "entropy {:.4} too low (max {:.4})",
            entropy,
            max_entropy,
        );
    }

    #[test]
    fn test_rejection_sample() {
        let mut rng = TauRng::new(42);
        let samples = rng.rejection_sample(5, 100);
        assert_eq!(samples.len(), 5);
        // Check no duplicates
        for i in 0..samples.len() {
            for j in (i + 1)..samples.len() {
                assert_ne!(samples[i], samples[j]);
            }
        }
        // Check range
        for &s in &samples {
            assert!(s >= 0 && s < 100);
        }
    }

    #[test]
    fn test_xoshiro_deterministic() {
        let mut rng1 = Xoshiro256StarStar::seed_from_u64(42);
        let mut rng2 = Xoshiro256StarStar::seed_from_u64(42);
        for _ in 0..100 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    #[test]
    fn test_xoshiro_f32_range() {
        let mut rng = Xoshiro256StarStar::seed_from_u64(42);
        for _ in 0..10_000 {
            let val = rng.random_f32();
            assert!(
                (0.0..1.0).contains(&val),
                "random_f32 out of range: {}",
                val
            );
        }
    }

    #[test]
    fn test_xoshiro_f64_range() {
        let mut rng = Xoshiro256StarStar::seed_from_u64(42);
        for _ in 0..10_000 {
            let val = rng.random_f64();
            assert!(
                (0.0..1.0).contains(&val),
                "random_f64 out of range: {}",
                val
            );
        }
    }

    #[test]
    fn test_xoshiro_range_i64() {
        let mut rng = Xoshiro256StarStar::seed_from_u64(42);
        for _ in 0..10_000 {
            let val = rng.random_range_i64(-100, 100);
            assert!(
                (-100..100).contains(&val),
                "random_range_i64 out of range: {}",
                val
            );
        }
    }
}
