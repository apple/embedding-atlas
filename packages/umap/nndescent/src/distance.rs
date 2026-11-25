//! Distance metric implementations based on PyNNDescent (https://github.com/lmcinnes/pynndescent).
//! All functions take two `&[f32]` slices and return `f32`.

pub const FLOAT32_EPS: f32 = f32::EPSILON;
pub const FLOAT32_MAX: f32 = f32::MAX;

pub type DistanceFunc = fn(&[f32], &[f32]) -> f32;
pub type CorrectionFunc = fn(f32) -> f32;

// ===== Core distance functions =====

pub fn euclidean(x: &[f32], y: &[f32]) -> f32 {
    squared_euclidean(x, y).sqrt()
}

pub fn squared_euclidean(x: &[f32], y: &[f32]) -> f32 {
    kernel::squared_euclidean(x, y)
}

pub fn manhattan(x: &[f32], y: &[f32]) -> f32 {
    let mut result = 0.0f32;
    for i in 0..x.len() {
        result += (x[i] - y[i]).abs();
    }
    result
}

pub fn chebyshev(x: &[f32], y: &[f32]) -> f32 {
    let mut result = 0.0f32;
    for i in 0..x.len() {
        result = result.max((x[i] - y[i]).abs());
    }
    result
}

pub fn minkowski(x: &[f32], y: &[f32]) -> f32 {
    minkowski_p(x, y, 2.0)
}

pub fn minkowski_p(x: &[f32], y: &[f32], p: f32) -> f32 {
    let mut result = 0.0f32;
    for i in 0..x.len() {
        result += (x[i] - y[i]).abs().powf(p);
    }
    result.powf(1.0 / p)
}

pub fn cosine(x: &[f32], y: &[f32]) -> f32 {
    let (dot, norm_x, norm_y) = kernel::dot_norms(x, y);
    if norm_x == 0.0 && norm_y == 0.0 {
        0.0
    } else if norm_x == 0.0 || norm_y == 0.0 {
        1.0
    } else {
        1.0 - (dot / (norm_x * norm_y).sqrt())
    }
}

pub fn alternative_cosine(x: &[f32], y: &[f32]) -> f32 {
    let (dot, norm_x, norm_y) = kernel::dot_norms(x, y);
    if norm_x == 0.0 && norm_y == 0.0 {
        0.0
    } else if norm_x == 0.0 || norm_y == 0.0 || dot <= 0.0 {
        FLOAT32_MAX
    } else {
        let ratio = (norm_x * norm_y).sqrt() / dot;
        ratio.log2()
    }
}

/// `-log2(<x, y>)`, the fast cosine proxy for rows that are already
/// L2-normalized (PyNNDescent's `alternative_dot`). Corrected with
/// [`correct_alternative_cosine`]. Clamped at 0 because rounding can push the
/// dot product of near-identical unit vectors slightly above 1.
pub fn alternative_dot(x: &[f32], y: &[f32]) -> f32 {
    let dot = kernel::dot(x, y);
    if dot <= 0.0 {
        FLOAT32_MAX
    } else {
        (-dot.log2()).max(0.0)
    }
}

pub fn correlation(x: &[f32], y: &[f32]) -> f32 {
    let n = x.len() as f32;
    let mut mu_x = 0.0f32;
    let mut mu_y = 0.0f32;
    for i in 0..x.len() {
        mu_x += x[i];
        mu_y += y[i];
    }
    mu_x /= n;
    mu_y /= n;

    let mut norm_x = 0.0f32;
    let mut norm_y = 0.0f32;
    let mut dot_product = 0.0f32;
    for i in 0..x.len() {
        let sx = x[i] - mu_x;
        let sy = y[i] - mu_y;
        norm_x += sx * sx;
        norm_y += sy * sy;
        dot_product += sx * sy;
    }
    if norm_x == 0.0 && norm_y == 0.0 {
        0.0
    } else if dot_product == 0.0 {
        1.0
    } else {
        1.0 - (dot_product / (norm_x * norm_y).sqrt())
    }
}

pub fn hamming(x: &[f32], y: &[f32]) -> f32 {
    let mut result = 0.0f32;
    for i in 0..x.len() {
        if x[i] != y[i] {
            result += 1.0;
        }
    }
    result / x.len() as f32
}

pub fn canberra(x: &[f32], y: &[f32]) -> f32 {
    let mut result = 0.0f32;
    for i in 0..x.len() {
        let denom = x[i].abs() + y[i].abs();
        if denom > 0.0 {
            result += (x[i] - y[i]).abs() / denom;
        }
    }
    result
}

pub fn bray_curtis(x: &[f32], y: &[f32]) -> f32 {
    let mut numerator = 0.0f32;
    let mut denominator = 0.0f32;
    for i in 0..x.len() {
        numerator += (x[i] - y[i]).abs();
        denominator += (x[i] + y[i]).abs();
    }
    if denominator > 0.0 {
        numerator / denominator
    } else {
        0.0
    }
}

// ===== Binary/set distance functions =====

pub fn jaccard(x: &[f32], y: &[f32]) -> f32 {
    let mut num_non_zero = 0.0f32;
    let mut num_equal = 0.0f32;
    for i in 0..x.len() {
        let x_true = x[i] != 0.0;
        let y_true = y[i] != 0.0;
        if x_true || y_true {
            num_non_zero += 1.0;
        }
        if x_true && y_true {
            num_equal += 1.0;
        }
    }
    if num_non_zero == 0.0 {
        0.0
    } else {
        (num_non_zero - num_equal) / num_non_zero
    }
}

pub fn alternative_jaccard(x: &[f32], y: &[f32]) -> f32 {
    let mut num_non_zero = 0.0f32;
    let mut num_equal = 0.0f32;
    for i in 0..x.len() {
        let x_true = x[i] != 0.0;
        let y_true = y[i] != 0.0;
        if x_true || y_true {
            num_non_zero += 1.0;
        }
        if x_true && y_true {
            num_equal += 1.0;
        }
    }
    if num_non_zero == 0.0 {
        0.0
    } else {
        -(num_equal / num_non_zero).log2()
    }
}

pub fn dice(x: &[f32], y: &[f32]) -> f32 {
    let mut num_true_true = 0.0f32;
    let mut num_not_equal = 0.0f32;
    for i in 0..x.len() {
        let x_true = x[i] != 0.0;
        let y_true = y[i] != 0.0;
        if x_true && y_true {
            num_true_true += 1.0;
        }
        if x_true != y_true {
            num_not_equal += 1.0;
        }
    }
    if num_not_equal == 0.0 {
        0.0
    } else {
        num_not_equal / (2.0 * num_true_true + num_not_equal)
    }
}

pub fn matching(x: &[f32], y: &[f32]) -> f32 {
    let mut num_not_equal = 0.0f32;
    for i in 0..x.len() {
        let x_true = x[i] != 0.0;
        let y_true = y[i] != 0.0;
        if x_true != y_true {
            num_not_equal += 1.0;
        }
    }
    num_not_equal / x.len() as f32
}

pub fn rogers_tanimoto(x: &[f32], y: &[f32]) -> f32 {
    let mut num_not_equal = 0.0f32;
    for i in 0..x.len() {
        let x_true = x[i] != 0.0;
        let y_true = y[i] != 0.0;
        if x_true != y_true {
            num_not_equal += 1.0;
        }
    }
    (2.0 * num_not_equal) / (x.len() as f32 + num_not_equal)
}

pub fn russellrao(x: &[f32], y: &[f32]) -> f32 {
    let mut num_true_true = 0.0f32;
    let mut num_x_true = 0.0f32;
    let mut num_y_true = 0.0f32;
    for i in 0..x.len() {
        let x_true = x[i] != 0.0;
        let y_true = y[i] != 0.0;
        if x_true && y_true {
            num_true_true += 1.0;
        }
        if x_true {
            num_x_true += 1.0;
        }
        if y_true {
            num_y_true += 1.0;
        }
    }
    if num_true_true == num_x_true && num_true_true == num_y_true {
        0.0
    } else {
        (x.len() as f32 - num_true_true) / x.len() as f32
    }
}

pub fn sokal_michener(x: &[f32], y: &[f32]) -> f32 {
    rogers_tanimoto(x, y)
}

pub fn sokal_sneath(x: &[f32], y: &[f32]) -> f32 {
    let mut num_true_true = 0.0f32;
    let mut num_not_equal = 0.0f32;
    for i in 0..x.len() {
        let x_true = x[i] != 0.0;
        let y_true = y[i] != 0.0;
        if x_true && y_true {
            num_true_true += 1.0;
        }
        if x_true != y_true {
            num_not_equal += 1.0;
        }
    }
    if num_not_equal == 0.0 {
        0.0
    } else {
        num_not_equal / (0.5 * num_true_true + num_not_equal)
    }
}

pub fn yule(x: &[f32], y: &[f32]) -> f32 {
    let mut num_true_true = 0.0f32;
    let mut num_true_false = 0.0f32;
    let mut num_false_true = 0.0f32;
    for i in 0..x.len() {
        let x_true = x[i] != 0.0;
        let y_true = y[i] != 0.0;
        if x_true && y_true {
            num_true_true += 1.0;
        }
        if x_true && !y_true {
            num_true_false += 1.0;
        }
        if !x_true && y_true {
            num_false_true += 1.0;
        }
    }
    let num_false_false = x.len() as f32 - num_true_true - num_true_false - num_false_true;

    if num_true_false == 0.0 || num_false_true == 0.0 {
        0.0
    } else {
        (2.0 * num_true_false * num_false_true)
            / (num_true_true * num_false_false + num_true_false * num_false_true)
    }
}

pub fn haversine(x: &[f32], y: &[f32]) -> f32 {
    let sin_lat = (0.5 * (x[0] - y[0])).sin();
    let sin_long = (0.5 * (x[1] - y[1])).sin();
    let result = (sin_lat * sin_lat + x[0].cos() * y[0].cos() * sin_long * sin_long).sqrt();
    2.0 * result.asin()
}

pub fn hellinger(x: &[f32], y: &[f32]) -> f32 {
    let mut result = 0.0f32;
    let mut l1_norm_x = 0.0f32;
    let mut l1_norm_y = 0.0f32;
    for i in 0..x.len() {
        result += (x[i] * y[i]).sqrt();
        l1_norm_x += x[i];
        l1_norm_y += y[i];
    }
    if l1_norm_x == 0.0 && l1_norm_y == 0.0 {
        0.0
    } else if l1_norm_x == 0.0 || l1_norm_y == 0.0 {
        1.0
    } else {
        (1.0 - result / (l1_norm_x * l1_norm_y).sqrt())
            .max(0.0)
            .sqrt()
    }
}

pub fn alternative_hellinger(x: &[f32], y: &[f32]) -> f32 {
    let mut result = 0.0f32;
    let mut l1_norm_x = 0.0f32;
    let mut l1_norm_y = 0.0f32;
    for i in 0..x.len() {
        result += (x[i] * y[i]).sqrt();
        l1_norm_x += x[i];
        l1_norm_y += y[i];
    }
    if l1_norm_x == 0.0 && l1_norm_y == 0.0 {
        0.0
    } else if l1_norm_x == 0.0 || l1_norm_y == 0.0 || result <= 0.0 {
        FLOAT32_MAX
    } else {
        ((l1_norm_x * l1_norm_y).sqrt() / result).log2()
    }
}

// ===== Bit-packed distance functions (for u8 data) =====

pub fn bit_hamming(x: &[f32], y: &[f32]) -> f32 {
    // x and y are u8 data stored as f32 (cast to u8 for bit operations)
    let mut result = 0u32;
    for i in 0..x.len() {
        let xb = x[i] as u8;
        let yb = y[i] as u8;
        result += (xb ^ yb).count_ones();
    }
    result as f32
}

pub fn bit_jaccard(x: &[f32], y: &[f32]) -> f32 {
    let mut num_or = 0u32;
    let mut num_and = 0u32;
    for i in 0..x.len() {
        let xb = x[i] as u8;
        let yb = y[i] as u8;
        num_or += (xb | yb).count_ones();
        num_and += (xb & yb).count_ones();
    }
    if num_or == 0 {
        0.0
    } else {
        let jaccard_sim = num_and as f64 / num_or as f64;
        if jaccard_sim > 0.0 {
            -(jaccard_sim.ln()) as f32
        } else {
            FLOAT32_MAX
        }
    }
}

// Bit-packed versions that work with &[u8] directly
pub fn bit_hamming_u8(x: &[u8], y: &[u8]) -> f32 {
    let mut result = 0u32;
    for i in 0..x.len() {
        result += (x[i] ^ y[i]).count_ones();
    }
    result as f32
}

pub fn bit_jaccard_u8(x: &[u8], y: &[u8]) -> f32 {
    let mut num_or = 0u32;
    let mut num_and = 0u32;
    for i in 0..x.len() {
        num_or += (x[i] | y[i]).count_ones();
        num_and += (x[i] & y[i]).count_ones();
    }
    if num_or == 0 {
        0.0
    } else {
        let jaccard_sim = num_and as f64 / num_or as f64;
        if jaccard_sim > 0.0 {
            -(jaccard_sim.ln()) as f32
        } else {
            FLOAT32_MAX
        }
    }
}

// ===== Correction functions for fast alternatives =====

pub fn correct_squared_euclidean(d: f32) -> f32 {
    d.sqrt()
}

pub fn correct_alternative_cosine(d: f32) -> f32 {
    1.0 - 2.0_f32.powf(-d)
}

pub fn correct_alternative_jaccard(d: f32) -> f32 {
    1.0 - 2.0_f32.powf(-d)
}

pub fn correct_alternative_hellinger(d: f32) -> f32 {
    (1.0 - 2.0_f32.powf(-d)).max(0.0).sqrt()
}

// ===== Registry =====

pub fn get_distance_func(name: &str) -> Option<DistanceFunc> {
    match name {
        "euclidean" | "l2" => Some(euclidean),
        "sqeuclidean" | "squared_euclidean" => Some(squared_euclidean),
        "manhattan" | "taxicab" | "l1" => Some(manhattan),
        "chebyshev" | "infinity" | "linfinity" | "linfty" | "linf" => Some(chebyshev),
        "minkowski" => Some(minkowski),
        "cosine" => Some(cosine),
        "correlation" => Some(correlation),
        "hamming" => Some(hamming),
        "jaccard" => Some(jaccard),
        "dice" => Some(dice),
        "matching" => Some(matching),
        "rogerstanimoto" => Some(rogers_tanimoto),
        "russellrao" => Some(russellrao),
        "sokalmichener" => Some(sokal_michener),
        "sokalsneath" => Some(sokal_sneath),
        "yule" => Some(yule),
        "canberra" => Some(canberra),
        "braycurtis" => Some(bray_curtis),
        "haversine" => Some(haversine),
        "hellinger" => Some(hellinger),
        "bit_hamming" => Some(bit_hamming),
        "bit_jaccard" => Some(bit_jaccard),
        "dot" => Some(cosine), // dot uses cosine on normalized data
        _ => None,
    }
}

/// Returns (fast_distance_func, correction_func) for metrics that have optimized alternatives
pub fn get_fast_alternative(name: &str) -> Option<(DistanceFunc, CorrectionFunc)> {
    match name {
        "euclidean" | "l2" => Some((squared_euclidean, correct_squared_euclidean)),
        "cosine" | "dot" => Some((alternative_cosine, correct_alternative_cosine)),
        "hellinger" => Some((alternative_hellinger, correct_alternative_hellinger)),
        "jaccard" => Some((alternative_jaccard, correct_alternative_jaccard)),
        _ => None,
    }
}

/// Fast alternative for metrics that can run on L2-normalized rows instead
/// (`cosine`, `dot`). Only valid if the data, and any query vectors, have been
/// normalized first; otherwise use [`get_fast_alternative`].
pub fn get_normalized_fast_alternative(name: &str) -> Option<(DistanceFunc, CorrectionFunc)> {
    match name {
        "cosine" | "dot" => Some((alternative_dot, correct_alternative_cosine)),
        _ => None,
    }
}

/// Whether the metric uses angular (rather than euclidean) RP tree splitting
pub fn is_angular_metric(name: &str) -> bool {
    matches!(
        name,
        "cosine"
            | "dot"
            | "correlation"
            | "dice"
            | "jaccard"
            | "hellinger"
            | "hamming"
            | "bit_hamming"
            | "bit_jaccard"
    )
}

/// Whether the metric uses bit-packed data
pub fn is_bit_metric(name: &str) -> bool {
    matches!(name, "bit_hamming" | "bit_jaccard")
}

/// Compute the L2 norm of a vector
pub fn norm(vec: &[f32]) -> f32 {
    let mut result = 0.0f32;
    for &v in vec {
        result += v * v;
    }
    result.sqrt()
}

/// Normalize a vector to unit length in-place. Returns the original norm.
pub fn normalize_vector(vec: &mut [f32]) -> f32 {
    let n = norm(vec);
    if n > 0.0 {
        for v in vec.iter_mut() {
            *v /= n;
        }
    }
    n
}

/// Inner-loop primitives shared by the float distances. On wasm32 with
/// simd128 these use direct `v128` loads and four independent accumulators
/// per sum (16 floats per step); elsewhere a portable `wide` implementation.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod kernel {
    use core::arch::wasm32::*;

    /// # Safety
    /// `p.add(i)..p.add(i + 4)` must be in bounds.
    #[inline(always)]
    unsafe fn load(p: *const f32, i: usize) -> v128 {
        // `v128_load` permits unaligned addresses.
        v128_load(p.add(i) as *const v128)
    }

    #[inline(always)]
    fn reduce(a: [v128; 4]) -> f32 {
        let v = f32x4_add(f32x4_add(a[0], a[1]), f32x4_add(a[2], a[3]));
        f32x4_extract_lane::<0>(v)
            + f32x4_extract_lane::<1>(v)
            + f32x4_extract_lane::<2>(v)
            + f32x4_extract_lane::<3>(v)
    }

    #[inline]
    pub fn squared_euclidean(x: &[f32], y: &[f32]) -> f32 {
        assert_eq!(x.len(), y.len());
        let n = x.len();
        let end = n - n % 16;
        let (px, py) = (x.as_ptr(), y.as_ptr());
        let mut acc = [f32x4_splat(0.0); 4];
        let mut i = 0;
        while i < end {
            for (l, a) in acc.iter_mut().enumerate() {
                // SAFETY: i + 16 <= end <= n == x.len() == y.len().
                let d = unsafe { f32x4_sub(load(px, i + 4 * l), load(py, i + 4 * l)) };
                *a = f32x4_add(*a, f32x4_mul(d, d));
            }
            i += 16;
        }
        let mut r = reduce(acc);
        for j in end..n {
            let d = x[j] - y[j];
            r += d * d;
        }
        r
    }

    #[inline]
    pub fn dot(x: &[f32], y: &[f32]) -> f32 {
        assert_eq!(x.len(), y.len());
        let n = x.len();
        let end = n - n % 16;
        let (px, py) = (x.as_ptr(), y.as_ptr());
        let mut acc = [f32x4_splat(0.0); 4];
        let mut i = 0;
        while i < end {
            for (l, a) in acc.iter_mut().enumerate() {
                // SAFETY: i + 16 <= end <= n == x.len() == y.len().
                let p = unsafe { f32x4_mul(load(px, i + 4 * l), load(py, i + 4 * l)) };
                *a = f32x4_add(*a, p);
            }
            i += 16;
        }
        let mut r = reduce(acc);
        for j in end..n {
            r += x[j] * y[j];
        }
        r
    }

    /// Returns `(<x, y>, |x|^2, |y|^2)`.
    #[inline]
    pub fn dot_norms(x: &[f32], y: &[f32]) -> (f32, f32, f32) {
        assert_eq!(x.len(), y.len());
        let n = x.len();
        let end = n - n % 16;
        let (px, py) = (x.as_ptr(), y.as_ptr());
        let z = f32x4_splat(0.0);
        let (mut d, mut nx, mut ny) = ([z; 4], [z; 4], [z; 4]);
        let mut i = 0;
        while i < end {
            for l in 0..4 {
                // SAFETY: i + 16 <= end <= n == x.len() == y.len().
                let (a, b) = unsafe { (load(px, i + 4 * l), load(py, i + 4 * l)) };
                d[l] = f32x4_add(d[l], f32x4_mul(a, b));
                nx[l] = f32x4_add(nx[l], f32x4_mul(a, a));
                ny[l] = f32x4_add(ny[l], f32x4_mul(b, b));
            }
            i += 16;
        }
        let (mut dot, mut norm_x, mut norm_y) = (reduce(d), reduce(nx), reduce(ny));
        for j in end..n {
            dot += x[j] * y[j];
            norm_x += x[j] * x[j];
            norm_y += y[j] * y[j];
        }
        (dot, norm_x, norm_y)
    }
}

#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
mod kernel {
    use wide::f32x4;

    #[inline(always)]
    fn load(v: &[f32], i: usize) -> f32x4 {
        f32x4::new([v[i], v[i + 1], v[i + 2], v[i + 3]])
    }

    #[inline]
    pub fn squared_euclidean(x: &[f32], y: &[f32]) -> f32 {
        assert_eq!(x.len(), y.len());
        let n = x.len();
        let mut acc0 = f32x4::ZERO;
        let mut acc1 = f32x4::ZERO;
        let end = n - n % 8;
        let mut i = 0;
        while i < end {
            let d0 = load(x, i) - load(y, i);
            acc0 = d0.mul_add(d0, acc0);
            let d1 = load(x, i + 4) - load(y, i + 4);
            acc1 = d1.mul_add(d1, acc1);
            i += 8;
        }
        let mut r = (acc0 + acc1).reduce_add();
        while i < n {
            let d = x[i] - y[i];
            r += d * d;
            i += 1;
        }
        r
    }

    #[inline]
    pub fn dot(x: &[f32], y: &[f32]) -> f32 {
        assert_eq!(x.len(), y.len());
        let n = x.len();
        let mut acc0 = f32x4::ZERO;
        let mut acc1 = f32x4::ZERO;
        let end = n - n % 8;
        let mut i = 0;
        while i < end {
            acc0 = load(x, i).mul_add(load(y, i), acc0);
            acc1 = load(x, i + 4).mul_add(load(y, i + 4), acc1);
            i += 8;
        }
        let mut r = (acc0 + acc1).reduce_add();
        while i < n {
            r += x[i] * y[i];
            i += 1;
        }
        r
    }

    /// Returns `(<x, y>, |x|^2, |y|^2)`.
    #[inline]
    pub fn dot_norms(x: &[f32], y: &[f32]) -> (f32, f32, f32) {
        assert_eq!(x.len(), y.len());
        let n = x.len();
        let (mut d0, mut d1) = (f32x4::ZERO, f32x4::ZERO);
        let (mut nx0, mut nx1) = (f32x4::ZERO, f32x4::ZERO);
        let (mut ny0, mut ny1) = (f32x4::ZERO, f32x4::ZERO);
        let end = n - n % 8;
        let mut i = 0;
        while i < end {
            let (xv0, yv0) = (load(x, i), load(y, i));
            d0 = xv0.mul_add(yv0, d0);
            nx0 = xv0.mul_add(xv0, nx0);
            ny0 = yv0.mul_add(yv0, ny0);
            let (xv1, yv1) = (load(x, i + 4), load(y, i + 4));
            d1 = xv1.mul_add(yv1, d1);
            nx1 = xv1.mul_add(xv1, nx1);
            ny1 = yv1.mul_add(yv1, ny1);
            i += 8;
        }
        let mut dot = (d0 + d1).reduce_add();
        let mut norm_x = (nx0 + nx1).reduce_add();
        let mut norm_y = (ny0 + ny1).reduce_add();
        while i < n {
            dot += x[i] * y[i];
            norm_x += x[i] * x[i];
            norm_y += y[i] * y[i];
            i += 1;
        }
        (dot, norm_x, norm_y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5
    }

    #[test]
    fn test_euclidean() {
        let x = [1.0f32, 2.0, 3.0];
        let y = [4.0f32, 5.0, 6.0];
        let d = euclidean(&x, &y);
        assert!(approx_eq(d, (27.0f32).sqrt()));
    }

    #[test]
    fn test_cosine() {
        let x = [1.0f32, 0.0, 0.0];
        let y = [0.0f32, 1.0, 0.0];
        assert!(approx_eq(cosine(&x, &y), 1.0));

        let x = [1.0f32, 0.0, 0.0];
        let y = [1.0f32, 0.0, 0.0];
        assert!(approx_eq(cosine(&x, &y), 0.0));
    }

    #[test]
    fn test_cosine_zero_vectors() {
        let x = [0.0f32; 3];
        let y = [0.0f32; 3];
        assert_eq!(cosine(&x, &y), 0.0);

        let x = [0.0f32; 3];
        let y = [1.0f32, 0.0, 0.0];
        assert_eq!(cosine(&x, &y), 1.0);
    }

    #[test]
    fn test_manhattan() {
        let x = [1.0f32, 2.0, 3.0];
        let y = [4.0f32, 5.0, 6.0];
        assert!(approx_eq(manhattan(&x, &y), 9.0));
    }

    #[test]
    fn test_jaccard_binary() {
        // [1, 1, 0] vs [1, 0, 1] -> intersection=1, union=3, dist=2/3
        let x = [1.0f32, 1.0, 0.0];
        let y = [1.0f32, 0.0, 1.0];
        assert!(approx_eq(jaccard(&x, &y), 2.0 / 3.0));
    }

    #[test]
    fn test_fast_alternative_euclidean() {
        let x = [1.0f32, 2.0, 3.0];
        let y = [4.0f32, 5.0, 6.0];
        let (fast_fn, correction) = get_fast_alternative("euclidean").unwrap();
        let fast_d = fast_fn(&x, &y);
        let corrected = correction(fast_d);
        assert!(approx_eq(corrected, euclidean(&x, &y)));
    }
}
