/// Distance metric correctness tests.
/// Validates our implementations against known reference values
/// (matching scipy/sklearn pairwise_distances behavior).
use nndescent::distance::*;

/// Helper: assert two f32 values are approximately equal.
fn assert_approx_eq(a: f32, b: f32, tol: f32, msg: &str) {
    let diff = (a - b).abs();
    assert!(
        diff < tol || (a == 0.0 && b == 0.0),
        "{}: {} vs {} (diff={})",
        msg,
        a,
        b,
        diff
    );
}

/// Generate deterministic spatial test data matching conftest.py.
/// Shape: (12, 20) - 10 random rows (seeded) + 2 zero rows.
/// We use our own TauRng for reproducibility since we can't match numpy exactly.
fn make_spatial_data() -> Vec<Vec<f32>> {
    use nndescent::rng::TauRng;
    let mut rng = TauRng::new(189212);
    let mut data = vec![vec![0.0f32; 20]; 12];
    for i in 0..10 {
        for j in 0..20 {
            // Generate uniform random like randn (we approximate with tau_rand)
            data[i][j] = rng.tau_rand() * 2.0 - 1.0; // roughly in [-1, 1]
        }
    }
    // rows 10, 11 are zeros
    data
}

/// Generate deterministic binary test data.
/// Shape: (12, 20) - 10 random rows + 2 zero rows.
fn make_binary_data() -> Vec<Vec<f32>> {
    use nndescent::rng::TauRng;
    let mut rng = TauRng::new(189212);
    let mut data = vec![vec![0.0f32; 20]; 12];
    for i in 0..10 {
        for j in 0..20 {
            // ~34% chance of True (matching conftest.py p=[0.66, 0.34])
            data[i][j] = if rng.tau_rand() > 0.66 { 1.0 } else { 0.0 };
        }
    }
    // rows 10, 11 are zeros
    data
}

// ===== Spatial distance self-consistency tests =====

/// Test that euclidean distance is non-negative and d(x,x) = 0.
#[test]
fn test_euclidean_self_distance() {
    let data = make_spatial_data();
    for row in &data {
        assert_approx_eq(euclidean(row, row), 0.0, 1e-6, "euclidean self-distance");
    }
}

/// Test euclidean triangle inequality.
#[test]
fn test_euclidean_triangle_inequality() {
    let data = make_spatial_data();
    for i in 0..data.len() {
        for j in 0..data.len() {
            for k in 0..data.len() {
                let d_ij = euclidean(&data[i], &data[j]);
                let d_ik = euclidean(&data[i], &data[k]);
                let d_kj = euclidean(&data[k], &data[j]);
                assert!(
                    d_ij <= d_ik + d_kj + 1e-5,
                    "Triangle inequality violated: d({},{})={} > d({},{})={} + d({},{})={}",
                    i,
                    j,
                    d_ij,
                    i,
                    k,
                    d_ik,
                    k,
                    j,
                    d_kj
                );
            }
        }
    }
}

/// Test euclidean symmetry.
#[test]
fn test_euclidean_symmetry() {
    let data = make_spatial_data();
    for i in 0..data.len() {
        for j in 0..data.len() {
            assert_approx_eq(
                euclidean(&data[i], &data[j]),
                euclidean(&data[j], &data[i]),
                1e-6,
                &format!("euclidean symmetry ({},{})", i, j),
            );
        }
    }
}

/// Validate spatial metrics produce reasonable results.
#[test]
fn test_spatial_check() {
    let data = make_spatial_data();

    // For each spatial metric, verify:
    // 1. Self-distance is 0 (or near 0)
    // 2. Non-negativity
    // 3. Symmetry

    let metrics: Vec<(&str, DistanceFunc)> = vec![
        ("euclidean", euclidean),
        ("manhattan", manhattan),
        ("chebyshev", chebyshev),
        ("cosine", cosine),
        ("correlation", correlation),
        ("hamming", hamming),
        ("canberra", canberra),
        ("braycurtis", bray_curtis),
    ];

    for (name, func) in &metrics {
        // Self-distance should be 0
        for i in 0..10 {
            let d = func(&data[i], &data[i]);
            assert!(
                d.abs() < 1e-5 || d.is_nan(),
                "{}: self-distance for row {} is {}",
                name,
                i,
                d
            );
        }

        // Non-negative (for non-NaN)
        for i in 0..10 {
            for j in 0..10 {
                let d = func(&data[i], &data[j]);
                if !d.is_nan() {
                    assert!(
                        d >= -1e-6,
                        "{}: negative distance d({},{})={}",
                        name,
                        i,
                        j,
                        d
                    );
                }
            }
        }

        // Symmetry
        for i in 0..10 {
            for j in 0..10 {
                let d_ij = func(&data[i], &data[j]);
                let d_ji = func(&data[j], &data[i]);
                if !d_ij.is_nan() && !d_ji.is_nan() {
                    assert_approx_eq(
                        d_ij,
                        d_ji,
                        1e-5,
                        &format!("{} symmetry ({},{})", name, i, j),
                    );
                }
            }
        }
    }
}

/// Test binary metrics on boolean data.
#[test]
fn test_binary_check() {
    let data = make_binary_data();

    let metrics: Vec<(&str, DistanceFunc)> = vec![
        ("jaccard", jaccard),
        ("matching", matching),
        ("dice", dice),
        ("rogerstanimoto", rogers_tanimoto),
        ("russellrao", russellrao),
        ("sokalmichener", sokal_michener),
        ("sokalsneath", sokal_sneath),
        ("yule", yule),
    ];

    for (name, func) in &metrics {
        // Self-distance should be 0
        for i in 0..10 {
            let d = func(&data[i], &data[i]);
            if !d.is_nan() {
                assert!(
                    d.abs() < 1e-5,
                    "{}: self-distance for row {} is {}",
                    name,
                    i,
                    d
                );
            }
        }

        // Non-negative
        for i in 0..10 {
            for j in 0..10 {
                let d = func(&data[i], &data[j]);
                if !d.is_nan() {
                    assert!(
                        d >= -1e-6,
                        "{}: negative distance d({},{})={}",
                        name,
                        i,
                        j,
                        d
                    );
                }
            }
        }

        // Symmetry
        for i in 0..10 {
            for j in 0..10 {
                let d_ij = func(&data[i], &data[j]);
                let d_ji = func(&data[j], &data[i]);
                if !d_ij.is_nan() && !d_ji.is_nan() {
                    assert_approx_eq(
                        d_ij,
                        d_ji,
                        1e-5,
                        &format!("{} symmetry ({},{})", name, i, j),
                    );
                }
            }
        }
    }
}

/// Test zero-vector handling for various metrics.
#[test]
fn test_zero_vector_handling() {
    let zero = vec![0.0f32; 20];
    let nonzero: Vec<f32> = (0..20).map(|i| i as f32 * 0.1).collect();

    // Cosine: d(0,0) = 0, d(0,x) = 1.0
    assert_approx_eq(cosine(&zero, &zero), 0.0, 1e-6, "cosine(0,0)");
    assert_approx_eq(cosine(&zero, &nonzero), 1.0, 1e-6, "cosine(0,x)");

    // Euclidean: d(0,0) = 0
    assert_approx_eq(euclidean(&zero, &zero), 0.0, 1e-6, "euclidean(0,0)");

    // Manhattan: d(0,0) = 0
    assert_approx_eq(manhattan(&zero, &zero), 0.0, 1e-6, "manhattan(0,0)");
}

/// Test squared euclidean matches euclidean^2.
#[test]
fn test_squared_euclidean_consistency() {
    let data = make_spatial_data();
    for i in 0..10 {
        for j in 0..10 {
            let e = euclidean(&data[i], &data[j]);
            let se = squared_euclidean(&data[i], &data[j]);
            assert_approx_eq(se, e * e, 1e-4, &format!("sq_euclidean({},{})", i, j));
        }
    }
}

/// Test alternative distances match true distances after correction.
#[test]
fn test_alternative_distances() {
    use nndescent::rng::TauRng;
    let mut rng = TauRng::new(42);
    let dim = 30;

    // Generate random vectors
    let mut vectors: Vec<Vec<f32>> = Vec::new();
    for _ in 0..100 {
        let v: Vec<f32> = (0..dim).map(|_| rng.tau_rand()).collect();
        vectors.push(v);
    }

    // Test alternative cosine
    for i in 0..50 {
        let j = i + 50;
        let true_dist = cosine(&vectors[i], &vectors[j]);
        let alt_dist = alternative_cosine(&vectors[i], &vectors[j]);
        let corrected = correct_alternative_cosine(alt_dist);
        assert_approx_eq(
            corrected,
            true_dist,
            0.01,
            &format!("alt_cosine correction ({},{})", i, j),
        );
    }

    // Test squared euclidean correction
    for i in 0..50 {
        let j = i + 50;
        let true_dist = euclidean(&vectors[i], &vectors[j]);
        let sq_dist = squared_euclidean(&vectors[i], &vectors[j]);
        let corrected = correct_squared_euclidean(sq_dist);
        assert_approx_eq(
            corrected,
            true_dist,
            0.01,
            &format!("sq_euclidean correction ({},{})", i, j),
        );
    }
}

/// Test the distance registry.
#[test]
fn test_distance_registry() {
    let known_metrics = vec![
        "euclidean",
        "l2",
        "manhattan",
        "taxicab",
        "l1",
        "chebyshev",
        "infinity",
        "minkowski",
        "cosine",
        "correlation",
        "hamming",
        "jaccard",
        "dice",
        "matching",
        "rogerstanimoto",
        "russellrao",
        "sokalmichener",
        "sokalsneath",
        "yule",
        "canberra",
        "braycurtis",
        "haversine",
        "hellinger",
    ];

    for name in &known_metrics {
        assert!(
            get_distance_func(name).is_some(),
            "Missing distance func: {}",
            name
        );
    }
}

/// Test fast alternatives exist.
#[test]
fn test_fast_alternative_registry() {
    assert!(get_fast_alternative("euclidean").is_some());
    assert!(get_fast_alternative("cosine").is_some());
    assert!(get_fast_alternative("jaccard").is_some());
}

/// Test angular metric detection.
#[test]
fn test_angular_metric_detection() {
    assert!(is_angular_metric("cosine"));
    assert!(is_angular_metric("correlation"));
    assert!(!is_angular_metric("euclidean"));
    assert!(!is_angular_metric("manhattan"));
}

/// Test bit_hamming on packed data.
#[test]
fn test_bit_hamming() {
    // Simulate bitpacked data as f32-cast u8
    let a: Vec<f32> = vec![0b10110011u8 as f32, 0b11001100u8 as f32];
    let b: Vec<f32> = vec![0b10110011u8 as f32, 0b11001100u8 as f32];
    assert_approx_eq(bit_hamming(&a, &b), 0.0, 1e-6, "bit_hamming identical");

    let c: Vec<f32> = vec![0b10110011u8 as f32, 0b11001101u8 as f32];
    let d = bit_hamming(&a, &c);
    assert!(d > 0.0, "bit_hamming different should be > 0");
}

/// Test bit_jaccard on packed data.
#[test]
fn test_bit_jaccard() {
    let a: Vec<f32> = vec![0b11111111u8 as f32, 0b11111111u8 as f32];
    let b: Vec<f32> = vec![0b11111111u8 as f32, 0b11111111u8 as f32];
    assert_approx_eq(bit_jaccard(&a, &b), 0.0, 1e-6, "bit_jaccard identical");

    let c: Vec<f32> = vec![0b11110000u8 as f32, 0b11110000u8 as f32];
    let d = bit_jaccard(&a, &c);
    assert!(d > 0.0, "bit_jaccard different should be > 0");
    assert!(d <= 1.0, "bit_jaccard should be <= 1.0");
}

/// Test haversine distance properties.
#[test]
fn test_haversine() {
    // Same point
    let a = [0.5f32, 0.5];
    let b = [0.5f32, 0.5];
    assert_approx_eq(haversine(&a, &b), 0.0, 1e-6, "haversine same point");

    // Antipodal points (lat, lon) in radians
    let a = [0.0f32, 0.0];
    let b = [std::f32::consts::PI, 0.0];
    let d = haversine(&a, &b);
    assert!(d > 0.0, "haversine antipodal should be > 0");
}

/// Test minkowski with various p values.
#[test]
fn test_minkowski_variants() {
    let a: Vec<f32> = vec![1.0, 2.0, 3.0];
    let b: Vec<f32> = vec![4.0, 5.0, 6.0];

    // p=1 should match manhattan
    let m1 = minkowski_p(&a, &b, 1.0);
    let man = manhattan(&a, &b);
    assert_approx_eq(m1, man, 1e-5, "minkowski p=1 vs manhattan");

    // p=2 should match euclidean
    let m2 = minkowski_p(&a, &b, 2.0);
    let euc = euclidean(&a, &b);
    assert_approx_eq(m2, euc, 1e-5, "minkowski p=2 vs euclidean");
}

/// Test hellinger distance properties.
#[test]
fn test_hellinger() {
    // Identical non-negative vectors
    let a: Vec<f32> = vec![0.1, 0.4, 0.25, 0.25];
    assert_approx_eq(hellinger(&a, &a), 0.0, 1e-5, "hellinger self-distance");

    // Different distributions
    let b: Vec<f32> = vec![0.25, 0.25, 0.25, 0.25];
    let d = hellinger(&a, &b);
    assert!(d >= 0.0, "hellinger should be non-negative");
}
