// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

mod array_io;
use density_clustering::*;
use serde::Serialize;
use std::collections::HashMap;

use clap::Parser;

/// Search for a pattern in a file and display the lines that contain it.
#[derive(Parser)]
struct Cli {
    /// Path to the density map binary file.
    path: String,

    /// The width of the density map.
    width: i32,

    /// The height of the density map.
    height: i32,

    /// Path to the output density map file.
    #[arg(short, long, default_value = "clusters.bin")]
    output: String,

    /// Path to the output json file.
    #[arg(long, default_value = "clusters.json")]
    output_json: String,

    #[arg(long)]
    options: Option<String>,
}

#[derive(Serialize)]
struct OutputJSON {
    summaries: HashMap<i32, ClusterSummary>,
    boundaries: HashMap<i32, Vec<Vec<(f64, f64)>>>,
}

pub fn main() {
    let t0 = std::time::Instant::now();

    let args = Cli::parse();
    let density_map =
        array_io::read_f32_array(args.path.as_str(), args.width, args.height).unwrap();

    let options = match args.options {
        Some(json) => serde_json::from_str(json.as_str()).unwrap(),
        None => FindClustersOptions::default(),
    };

    let t1 = std::time::Instant::now();

    let (cluster_map, clusters) = find_clusters(&density_map, &options);

    let boundaries: HashMap<i32, Vec<Vec<(f64, f64)>>> = trace_all_outer_contours(&cluster_map)
        .iter()
        .map(|(&cluster_id, polygons)| {
            let polygons_f: Vec<Vec<(f64, f64)>> = polygons
                .iter()
                .map(|polygon| {
                    let mut polygon_f =
                        polygon.iter().map(|&(x, y)| (x as f64, y as f64)).collect();
                    polygon_f = smooth_polygon(&polygon_f);
                    polygon_f
                })
                .collect();
            (cluster_id, polygons_f)
        })
        .collect();

    let t2 = std::time::Instant::now();

    array_io::write_i32_array(&args.output, &cluster_map).unwrap();

    let file = std::fs::File::create(args.output_json).unwrap();
    let bw = std::io::BufWriter::new(file);
    serde_json::to_writer(
        bw,
        &OutputJSON {
            summaries: clusters,
            boundaries: boundaries,
        },
    )
    .unwrap();

    let t3 = std::time::Instant::now();

    println!(
        "load data: {:?}, algorithm: {:?}, save results: {:?}",
        t1 - t0,
        t2 - t1,
        t3 - t2,
    );
}
