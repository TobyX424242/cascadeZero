#[derive(Debug, Clone)]
pub(crate) struct SmallRng {
    state: u64,
}

impl SmallRng {
    pub(crate) fn new(seed: u64) -> Self {
        let seed = if seed == 0 {
            0x9e37_79b9_7f4a_7c15
        } else {
            seed
        };
        Self { state: seed }
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        let mut value = self.state;
        value ^= value >> 12;
        value ^= value << 25;
        value ^= value >> 27;
        self.state = value;
        value.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    pub(crate) fn next_f32(&mut self) -> f32 {
        let value = (self.next_u64() >> 40) as u32;
        value as f32 / (1u32 << 24) as f32
    }

    pub(crate) fn sample_weighted_index(&mut self, weights: &[f32]) -> usize {
        let total_weight: f32 = weights.iter().sum();
        if total_weight <= 0.0 {
            return (self.next_u64() as usize) % weights.len();
        }

        let mut threshold = self.next_f32() * total_weight;
        for (index, weight) in weights.iter().enumerate() {
            threshold -= *weight;
            if threshold <= 0.0 {
                return index;
            }
        }

        weights.len() - 1
    }

    pub(crate) fn sample_dirichlet(&mut self, alpha: f32, len: usize) -> Vec<f32> {
        if len == 0 {
            return Vec::new();
        }

        let alpha = alpha.max(1e-3);
        let mut samples = Vec::with_capacity(len);
        let mut total = 0.0;

        for _ in 0..len {
            let sample = self.sample_gamma(alpha);
            samples.push(sample);
            total += sample;
        }

        if !total.is_finite() || total <= 0.0 {
            let uniform = 1.0 / len as f32;
            return vec![uniform; len];
        }

        for sample in &mut samples {
            *sample /= total;
        }
        samples
    }

    fn sample_gamma(&mut self, shape: f32) -> f32 {
        if shape <= 0.0 {
            return 0.0;
        }

        if shape < 1.0 {
            let base = self.sample_gamma(shape + 1.0);
            let uniform = self.next_f32().clamp(1e-7, 1.0 - 1e-7);
            return base * uniform.powf(1.0 / shape);
        }

        let d = shape - 1.0 / 3.0;
        let c = (1.0 / (9.0 * d)).sqrt();

        loop {
            let x = self.sample_standard_normal();
            let v = 1.0 + c * x;
            if v <= 0.0 {
                continue;
            }

            let v_cubed = v * v * v;
            let uniform = self.next_f32().clamp(1e-7, 1.0 - 1e-7);
            if uniform < 1.0 - 0.0331 * x.powi(4) {
                return d * v_cubed;
            }

            if uniform.ln() < 0.5 * x * x + d * (1.0 - v_cubed + v_cubed.ln()) {
                return d * v_cubed;
            }
        }
    }

    fn sample_standard_normal(&mut self) -> f32 {
        let u1 = self.next_f32().clamp(1e-7, 1.0 - 1e-7);
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
}

pub(crate) fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use super::SmallRng;

    #[test]
    fn dirichlet_samples_are_normalized() {
        let mut rng = SmallRng::new(7);
        let samples = rng.sample_dirichlet(0.3, 8);
        let total: f32 = samples.iter().sum();

        assert_eq!(samples.len(), 8);
        assert!((total - 1.0).abs() < 1e-5);
        assert!(samples.iter().all(|value| *value >= 0.0));
    }
}
