// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

use std::f64::consts::PI;

pub fn smooth_polygon(polygon: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let coeffs = lowpass_fir_filter(1.0, 0.05, 59);
    let mut result = vec![(0.0, 0.0); polygon.len()];

    for (j, &c) in coeffs.iter().enumerate() {
        for (i, &pt) in polygon.iter().enumerate() {
            let p = (i + j) % result.len();
            result[p].0 += pt.0 * c;
            result[p].1 += pt.1 * c;
        }
    }

    result
}

fn sinc(x: f64) -> f64 {
    if x != 0.0 {
        (x * PI).sin() / (x * PI)
    } else {
        1.0
    }
}

fn blackman(i: usize, length: usize) -> f64 {
    let p = (i as f64) / ((length - 1) as f64) * PI * 2.0;
    0.42 - 0.5 * p.cos() + 0.08 * (p * 2.0).cos()
}

fn lowpass_fir_filter(sample_rate: f64, cutoff_frequency: f64, length: usize) -> Vec<f64> {
    let mut x: Vec<f64> = (0..length)
        .map(|i| {
            let t = (i as f64) - ((length - 1) as f64) / 2.0;
            sinc(2.0 * cutoff_frequency * t / sample_rate) * blackman(i, length)
        })
        .collect();
    let x_sum: f64 = x.iter().sum();
    x.iter_mut().for_each(|x| *x /= x_sum);
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_low_pass_fir_filter() {
        let filter = lowpass_fir_filter(1.0, 0.1, 59);
        println!("{:?}", filter);
    }
}
