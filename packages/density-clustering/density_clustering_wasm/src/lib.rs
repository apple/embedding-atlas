use density_clustering::{smooth_polygon, Array2D, ClusterSummary};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(typescript_custom_section)]
const TS_JAVASCRIPT_CALLBACKS: &'static str = r#"
export interface ClusteringOptions {

}

export interface FindClustersOptions {
    clustering_options?: Partial<ClusteringOptions>;
    smooth_boundaries?: boolean;
    return_boundary_rects?: boolean;
}

export interface ClusterSummary {
    num_pixels: number;
    sum_x_density: number;
    sum_y_density: number;
    sum_density: number;
    max_density: number;
    max_density_location: [number, number];
}

export type Polygon = [number, number][];
export type Rect = [number, number, number, number];

export interface FindClustersResult {
  summaries: Map<number, ClusterSummary>;
  boundaries: Map<number, Polygon[]>;
  boundary_rects: Map<number, Rect[]>;
}
"#;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(typescript_type = "FindClustersResult")]
    pub type IFindClustersResult;

    #[wasm_bindgen(typescript_type = "FindClustersOptions")]
    pub type IFindClustersOptions;
}

#[wasm_bindgen]
pub struct DensityMap {
    density_map: Array2D<f32>,
}

#[wasm_bindgen]
impl DensityMap {
    #[wasm_bindgen(constructor)]
    pub fn new(width: i32, height: i32, data: &[f32]) -> DensityMap {
        DensityMap {
            density_map: Array2D::new(width, height, data.to_vec()),
        }
    }

    pub fn width(&self) -> i32 {
        self.density_map.width()
    }

    pub fn height(&self) -> i32 {
        self.density_map.height()
    }
}

#[derive(Deserialize, Default)]
#[serde(default)]
pub struct FindClustersOptions {
    clustering_options: density_clustering::FindClustersOptions,
    smooth_boundaries: bool,
    return_boundary_rects: bool,
}

#[derive(Serialize)]
pub struct FindClustersResult {
    summaries: HashMap<i32, ClusterSummary>,
    boundaries: HashMap<i32, Vec<Vec<(f64, f64)>>>,
    boundary_rects: Option<HashMap<i32, Vec<(f64, f64, f64, f64)>>>,
}

fn map_values<K, V1, V2, F: Fn(&V1) -> V2>(map: &HashMap<K, V1>, func: F) -> HashMap<K, V2>
where
    K: std::cmp::Eq + std::hash::Hash + Copy,
{
    map.iter().map(|(key, value)| (*key, func(value))).collect()
}

#[wasm_bindgen]
pub fn find_clusters(input: &DensityMap, options: IFindClustersOptions) -> IFindClustersResult {
    let options: FindClustersOptions =
        serde_wasm_bindgen::from_value(options.obj).unwrap_or_default();
    let (cluster_map, summaries) =
        density_clustering::find_clusters(&input.density_map, &options.clustering_options);
    let all_contours = density_clustering::trace_all_outer_contours(&cluster_map);
    let boundaries = map_values(&all_contours, |polygons| {
        let polygons_f: Vec<Vec<(f64, f64)>> = polygons
            .iter()
            .map(|polygon| {
                let mut polygon_f: Vec<(f64, f64)> =
                    polygon.iter().map(|&(x, y)| (x as f64, y as f64)).collect();
                if options.smooth_boundaries {
                    polygon_f = smooth_polygon(&polygon_f);
                }
                polygon_f
            })
            .collect();
        polygons_f
    });
    let boundary_rects = if options.return_boundary_rects {
        Some(map_values(&boundaries, |polygons| {
            density_clustering::fit_rects_from_polygons(polygons.as_slice())
        }))
    } else {
        None
    };

    serde_wasm_bindgen::to_value(&FindClustersResult {
        summaries,
        boundaries,
        boundary_rects,
    })
    .unwrap()
    .into()
}
