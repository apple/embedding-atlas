// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, VecDeque};

use crate::array_2d::Array2D;
use crate::disjoint_set_2d::DisjointSet2D;
use crate::find_local_maxima::find_local_maxima;

struct QueueItem {
    location: (i32, i32),
    density: f32,
    cluster: i32,
}

impl PartialEq for QueueItem {
    fn eq(&self, other: &QueueItem) -> bool {
        self.location == other.location
    }
}
impl Eq for QueueItem {}

impl Ord for QueueItem {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.location == other.location {
            return Ordering::Equal;
        }
        if self.density > other.density {
            Ordering::Greater
        } else {
            Ordering::Less
        }
    }
}
impl PartialOrd for QueueItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

trait MinMaxUpdatable<T> {
    fn update_max(&mut self, value: &(T, f32));
    fn update_min(&mut self, value: &(T, f32));
}

impl<T> MinMaxUpdatable<T> for Option<(T, f32)>
where
    T: Copy,
{
    fn update_max(&mut self, value: &(T, f32)) {
        match self {
            Some(existing) => {
                if existing.1 < value.1 {
                    *existing = (value.0, value.1);
                }
            }
            None => *self = Some((value.0, value.1)),
        }
    }
    fn update_min(&mut self, value: &(T, f32)) {
        match self {
            Some(existing) => {
                if existing.1 > value.1 {
                    *existing = (value.0, value.1);
                }
            }
            None => *self = Some((value.0, value.1)),
        }
    }
}

#[derive(Debug, Copy, Clone, Serialize)]
pub struct ClusterSummary {
    pub num_pixels: usize,
    pub sum_density: f32,
    pub sum_x_density: f32,
    pub sum_y_density: f32,
    pub max_density: f32,
    pub max_density_location: (i32, i32),
}

impl ClusterSummary {
    fn zero() -> ClusterSummary {
        ClusterSummary {
            sum_density: 0.0,
            max_density: 0.0,
            sum_x_density: 0.0,
            sum_y_density: 0.0,
            max_density_location: (-1, -1),
            num_pixels: 0,
        }
    }

    fn update(&mut self, location: (i32, i32), density: f32) {
        if density > self.max_density {
            self.max_density = density;
            self.max_density_location = location;
        }
        self.sum_density += density;
        self.sum_x_density += density * location.0 as f32;
        self.sum_y_density += density * location.1 as f32;
        self.num_pixels += 1;
    }

    fn update_with_summary(&mut self, other: &Self) {
        if other.max_density > self.max_density {
            self.max_density = other.max_density;
            self.max_density_location = other.max_density_location;
        }
        self.sum_density += other.sum_density;
        self.sum_x_density += other.sum_x_density;
        self.sum_y_density += other.sum_y_density;
        self.num_pixels += other.num_pixels;
    }
}

fn find_initial_clusters(density_map: &Array2D<f32>) -> (Array2D<i32>, Vec<ClusterSummary>) {
    let width = density_map.width();
    let height = density_map.height();

    // Map each pixel to a cluster id. -1 means no cluster.
    let mut cluster_map = Array2D::<i32>::new_with_constant(width, height, -1);

    // Find all local maxima.
    let local_maxima = find_local_maxima(density_map);

    // Initialize BFS seeds and clusters with the local maxima.
    let mut heap = BinaryHeap::<QueueItem>::new();
    let mut clusters = Vec::<ClusterSummary>::new();

    for location in local_maxima {
        let cluster = clusters.len() as i32;
        clusters.push(ClusterSummary::zero());
        heap.push(QueueItem {
            location,
            density: density_map[location],
            cluster,
        });
        cluster_map[location] = cluster;
    }

    // BFS search directions.
    let directions = [(-1, 0), (1, 0), (0, -1), (0, 1)];

    let mut max_heap_size: usize = 0;

    // BFS main loop.
    while let Some(start) = heap.pop() {
        let (x, y) = start.location;
        let cluster = start.cluster;
        for (dx, dy) in directions {
            let (px, py) = (x + dx, y + dy);
            if px < 0 || px >= width || py < 0 || py >= height {
                continue;
            }
            if cluster_map[(px, py)] != -1 {
                continue;
            }
            // If the neighbor is not visited (not assigned a cluster),
            // then we add it to the queue if its density is lower.
            let density = density_map[(px, py)];
            if density > start.density {
                continue;
            }
            cluster_map[(px, py)] = cluster;
            heap.push(QueueItem {
                location: (px, py),
                density,
                cluster,
            });
        }
        // Update cluster info.
        clusters[cluster as usize].update(start.location, start.density);
        max_heap_size = max_heap_size.max(heap.len());
    }

    (cluster_map, clusters)
}

fn find_unlabeled_clusters(
    density_map: &Array2D<f32>,
    cluster_map: &mut Array2D<i32>,
    clusters: &mut Vec<ClusterSummary>,
) {
    let width = density_map.width();
    let height = density_map.height();

    let directions = [(-1, 0), (1, 0), (0, -1), (0, 1)];

    for (x0, y0) in density_map.iter_coords() {
        if cluster_map[(x0, y0)] != -1 {
            continue;
        }
        let cluster = clusters.len() as i32;
        let mut summary = ClusterSummary::zero();

        // BFS from the start point.
        let mut queue = VecDeque::<(i32, i32)>::new();
        queue.push_back((x0, y0));

        while let Some((x, y)) = queue.pop_front() {
            for (dx, dy) in directions {
                let (px, py) = (x + dx, y + dy);
                if px < 0 || px >= width || py < 0 || py >= height {
                    continue;
                }
                if cluster_map[(px, py)] != -1 {
                    continue;
                }
                cluster_map[(px, py)] = cluster;
                queue.push_back((px, py));
            }
            summary.update((x, y), density_map[(x, y)]);
        }

        clusters.push(summary);
    }
}

fn find_initial_clusters_disjoint_set(
    density_map: &Array2D<f32>,
) -> (Array2D<i32>, Vec<ClusterSummary>) {
    let width = density_map.width();
    let height = density_map.height();

    // Map each pixel to a cluster id. -1 means no cluster.
    let mut cluster_map = Array2D::<i32>::new_with_constant(width, height, -1);
    let mut disjoint_set = DisjointSet2D::new(width, height);
    let directions = [(-1, 0), (1, 0), (0, -1), (0, 1)];
    for (x, y) in density_map.iter_coords() {
        let mut max_neighbor: Option<((i32, i32), f32)> = None;
        let d = density_map[(x, y)];
        for (dx, dy) in directions {
            let (nx, ny) = (x + dx, y + dy);
            if nx < 0 || ny < 0 || nx >= width || ny >= height {
                continue;
            }
            let nd = density_map[(nx, ny)];
            if nd >= d {
                max_neighbor.update_max(&((nx, ny), nd));
            }
        }
        if let Some(((nx, ny), _)) = max_neighbor {
            disjoint_set.union((x, y), (nx, ny));
        }
    }

    let mut summaries = Vec::<ClusterSummary>::new();

    for (x, y) in density_map.iter_coords() {
        let p = disjoint_set.find_parent((x, y));
        if p == (x, y) && cluster_map[p] == -1 {
            cluster_map[p] = summaries.len() as i32;
            summaries.push(ClusterSummary::zero());
        }
    }
    for (x, y) in density_map.iter_coords() {
        let p = disjoint_set.find_parent((x, y));
        let id = cluster_map[p];
        cluster_map[(x, y)] = id;
        summaries[id as usize].update((x, y), density_map[(x, y)]);
    }

    (cluster_map, summaries)
}

#[derive(Clone)]
struct EdgeSummary {
    max_density: f32,
    pixels: Vec<(i32, i32, f32)>,
}

impl Default for EdgeSummary {
    fn default() -> EdgeSummary {
        EdgeSummary {
            max_density: 0.0,
            pixels: Vec::new(),
        }
    }
}

impl EdgeSummary {
    fn update_with_pixel(&mut self, location: (i32, i32), density: f32) {
        if density > self.max_density {
            self.max_density = density;
        }
        self.pixels.push((location.0, location.1, density));
    }

    fn update(&mut self, other: EdgeSummary) {
        self.max_density = f32::max(self.max_density, other.max_density);
        self.pixels.extend(other.pixels);
    }
}

struct ClusterGraph {
    /// Maps a node id to a map of its neighbors to edge weights.
    neighbor_map: HashMap<i32, HashMap<i32, EdgeSummary>>,
    /// Maps a node id to its cluster ids.
    cluster_ids: HashMap<i32, Vec<i32>>,
    /// Maps a node id to the cluster summary.
    summary_map: HashMap<i32, ClusterSummary>,
    /// Maps a node id to the minimum distance from its max density location to an edge.
    min_edge_distance_cache: HashMap<i32, Option<(i32, f32)>>,
}

fn location_distance(loc1: (i32, i32), loc2: (i32, i32)) -> f32 {
    let lsqr = (loc1.0 - loc2.0) * (loc1.0 - loc2.0) + (loc1.1 - loc2.1) * (loc1.1 - loc2.1);
    (lsqr as f32).sqrt()
}

impl ClusterGraph {
    fn new_with_cluster_map(
        density_map: &Array2D<f32>,
        cluster_map: &Array2D<i32>,
        summaries: &[ClusterSummary],
    ) -> ClusterGraph {
        let width = density_map.width();
        let height = density_map.height();
        let mut border_map = HashMap::<(i32, i32), EdgeSummary>::new();

        for (x, y) in density_map.iter_coords() {
            for (nx, ny) in [(x + 1, y), (x, y + 1)] {
                if nx >= width - 1 || ny >= height - 1 {
                    continue;
                }
                let c0 = cluster_map[(x, y)];
                let c1 = cluster_map[(nx, ny)];
                if c0 != c1 && c0 != -1 && c1 != -1 {
                    border_map
                        .entry((c0, c1))
                        .or_default()
                        .update_with_pixel((x, y), density_map[(x, y)]);
                    border_map
                        .entry((c1, c0))
                        .or_default()
                        .update_with_pixel((nx, ny), density_map[(nx, ny)]);
                }
            }
        }

        let mut neighbor_map = HashMap::<i32, HashMap<i32, EdgeSummary>>::new();

        for ((from, to), value) in border_map {
            let m = neighbor_map.entry(from).or_default();
            m.insert(to, value);
        }

        let mut cluster_ids = HashMap::<i32, Vec<i32>>::new();
        let mut summary_map = HashMap::<i32, ClusterSummary>::new();

        for id in 0..summaries.len() as i32 {
            cluster_ids.insert(id, vec![id]);
            summary_map.insert(id, summaries[id as usize]);
            neighbor_map.entry(id).or_default();
        }

        ClusterGraph {
            neighbor_map,
            cluster_ids,
            summary_map,
            min_edge_distance_cache: HashMap::new(),
        }
    }

    fn node_ids(&self) -> impl Iterator<Item = i32> + '_ {
        self.neighbor_map.keys().copied()
    }

    fn neighbors(&self, node: i32) -> impl Iterator<Item = i32> + '_ {
        self.neighbor_map[&node].keys().copied()
    }

    fn cluster_ids(&self, node: i32) -> &Vec<i32> {
        &self.cluster_ids[&node]
    }

    fn max_edge_density(&self, node: i32) -> f32 {
        self.neighbor_map[&node]
            .values()
            .map(|s| s.max_density)
            .max_by(f32::total_cmp)
            .unwrap_or(0.0)
    }

    fn max_edge_density_for_pair(&self, node1: i32, node2: i32) -> f32 {
        let chain = self.neighbor_map[&node1]
            .iter()
            .chain(self.neighbor_map[&node2].iter());

        chain
            .filter_map(|(&n, w)| {
                if n != node1 && n != node2 {
                    Some(w.max_density)
                } else {
                    None
                }
            })
            .max_by(f32::total_cmp)
            .unwrap_or(0.0)
    }

    fn union(&mut self, node1: i32, node2: i32) -> i32 {
        if node1 == node2 {
            return node1;
        }
        // For each neighbor of node2.
        let n2_neighbors = self.neighbor_map.remove(&node2).unwrap();
        self.neighbor_map.get_mut(&node1).unwrap().remove(&node2);
        for (n2_neighbor, w) in n2_neighbors {
            if n2_neighbor == node1 {
                continue;
            }
            // Union the weight to node1's neighbors.
            let n1_neighbors = self.neighbor_map.get_mut(&node1).unwrap();
            n1_neighbors.entry(n2_neighbor).or_default().update(w);
            // Remove the back edge, and union the back edge weight to a back edge to node 1.
            let back_neighbors = self.neighbor_map.get_mut(&n2_neighbor).unwrap();
            if let Some(w) = back_neighbors.remove(&node2) {
                back_neighbors.entry(node1).or_default().update(w);
            }
        }
        // Combine the cluster_ids at node2 into node1.
        let node2_ids = self.cluster_ids.remove(&node2).unwrap().into_iter();
        let cluster_ids = self.cluster_ids.get_mut(&node1).unwrap();
        cluster_ids.extend(node2_ids);
        let n2_summary = self.summary_map.remove(&node2).unwrap();
        self.summary_map
            .get_mut(&node1)
            .unwrap()
            .update_with_summary(&n2_summary);

        self.min_edge_distance_cache.remove(&node1);
        for n in self.neighbors(node1).collect::<Vec<i32>>() {
            self.min_edge_distance_cache.remove(&n);
        }

        node1
    }

    fn compute_min_distance_to_edge(&self, node: i32) -> Option<(i32, f32)> {
        let max_density_location = self.summary_map[&node].max_density_location;
        let mut min_neighbor: Option<(i32, f32)> = None;
        for (&n, edge) in self.neighbor_map[&node].iter() {
            for (x, y, _) in edge.pixels.iter() {
                let d = location_distance((*x, *y), max_density_location);
                min_neighbor.update_min(&(n, d));
            }
        }
        min_neighbor
    }

    fn min_distance_to_edge(&mut self, node: i32) -> Option<(i32, f32)> {
        if self.min_edge_distance_cache.contains_key(&node) {
            self.min_edge_distance_cache[&node]
        } else {
            let r = self.compute_min_distance_to_edge(node);
            self.min_edge_distance_cache.insert(node, r);
            r
        }
    }
}

fn cluster_neighbor_map_grouping(graph: &mut ClusterGraph, max_density_scaler: f32) {
    loop {
        let mut max_pair: Option<((i32, i32), f32)> = None;
        for node in graph.node_ids() {
            let max_w = graph.max_edge_density(node);
            let max_d = graph.summary_map[&node].max_density;
            for neighbor in graph.neighbors(node) {
                let w = graph.neighbor_map[&node][&neighbor].max_density;
                let max_w2 = graph.max_edge_density_for_pair(node, neighbor);
                if max_w2 >= max_w {
                    // Can't actually lower the threshold.
                    continue;
                }
                let max_d2 = graph.summary_map[&neighbor].max_density;
                if w >= f32::max(max_d, max_d2) * max_density_scaler {
                    max_pair.update_max(&((node, neighbor), max_w2));
                }
            }
        }
        if let Some(((n1, n2), _)) = max_pair {
            graph.union(n1, n2);
        } else {
            break;
        }
    }
}

fn cluster_neighbor_map_union(graph: &mut ClusterGraph, threshold: f32) {
    loop {
        let mut min_pair: Option<((i32, i32), f32)> = None;
        for node in graph.node_ids().collect::<Vec<i32>>() {
            if let Some((edge, d)) = graph.min_distance_to_edge(node) {
                min_pair.update_min(&((node, edge), d));
            }
        }
        if let Some(((a, b), d)) = min_pair {
            if d < threshold {
                graph.union(a, b);
            } else {
                break;
            }
        } else {
            break;
        }
    }
}

#[derive(Serialize, Deserialize, Copy, Clone)]
#[serde(default)]
pub struct FindClustersOptions {
    use_disjoint_set: bool,
    add_unlabeled: bool,
    truncate_to_max_density: bool,
    perform_neighbor_map_grouping: bool,
    union_threshold: f32,
    threshold_scaler: f32,
    density_lowerbound_scaler: f32,
    density_upperbound_scaler: f32,
    tilted_threshold_plane: bool,
    grouping_density_scaler: f32,
}

impl Default for FindClustersOptions {
    fn default() -> Self {
        FindClustersOptions {
            use_disjoint_set: false,
            add_unlabeled: true,
            truncate_to_max_density: true,
            perform_neighbor_map_grouping: true,
            union_threshold: 10.0,
            threshold_scaler: 1.0,
            density_upperbound_scaler: 0.8,
            density_lowerbound_scaler: 0.4,
            tilted_threshold_plane: true,
            grouping_density_scaler: 0.8,
        }
    }
}

pub fn find_clusters(
    density_map: &Array2D<f32>,
    options: &FindClustersOptions,
) -> (Array2D<i32>, HashMap<i32, ClusterSummary>) {
    let add_unlabeled = options.add_unlabeled;
    let truncate_to_max_density = options.truncate_to_max_density;
    let perform_neighbor_map_grouping = options.perform_neighbor_map_grouping;
    let union_threshold = options.union_threshold;
    let grouping_density_scaler = options.grouping_density_scaler;
    let threshold_scaler = options.threshold_scaler;
    let density_lowerbound_scaler = options.density_lowerbound_scaler;
    let density_upperbound_scaler = options.density_upperbound_scaler;
    let tilted_threshold_plane = options.tilted_threshold_plane;
    let use_disjoint_set = options.use_disjoint_set;

    // Find initial clusters from local maxima.
    let (mut cluster_map, clusters) = if use_disjoint_set {
        find_initial_clusters_disjoint_set(density_map)
    } else {
        let (mut cluster_map, mut clusters) = find_initial_clusters(density_map);

        // Then treat left-over pieces as clusters as well.
        if add_unlabeled {
            find_unlabeled_clusters(density_map, &mut cluster_map, &mut clusters);
        }
        (cluster_map, clusters)
    };

    // Construct cluster graph and do neighbor grouping.
    let mut graph = ClusterGraph::new_with_cluster_map(density_map, &cluster_map, &clusters);

    if union_threshold > 0.0 {
        cluster_neighbor_map_union(&mut graph, union_threshold);
    }

    if perform_neighbor_map_grouping {
        cluster_neighbor_map_grouping(&mut graph, grouping_density_scaler);
    }

    // Find the max edge weight for each cluster,
    // and construct a map from original cluster ids to group ids.
    let mut post_map: Vec<Option<(i32, (f64, f64, f64), &ClusterSummary)>> =
        vec![None; clusters.len()];

    for (&cluster_id, values) in graph.cluster_ids.iter() {
        let mut s = EdgeSummary::default();
        for v in graph.neighbor_map[&cluster_id].values() {
            s.update(v.clone());
        }
        let (a, b, c) = if tilted_threshold_plane {
            crate::plane_fitting::estimate_density_cutoff_plane(&s.pixels)
        } else {
            (0.0, 0.0, s.max_density as f64)
        };

        for &v in values {
            let summary = graph.summary_map.get(&cluster_id).unwrap();
            post_map[v as usize] = Some((cluster_id, (a, b, c), summary));
        }
    }

    // Construct the final cluster map.
    for (x, y) in density_map.iter_coords() {
        let id = &mut cluster_map[(x, y)];
        if *id == -1 {
            continue;
        }
        if let Some((cluster_id, (sx, sy, sc), summary)) = post_map[*id as usize] {
            if truncate_to_max_density {
                // Base threshold calculated from the max density or tilted-plane max density.
                let base_threshold = (sx * (x as f64) + sy * (y as f64) + sc) as f32;

                // Apply scaler to the threshold,
                // then clamp it to a ratio of max_density.
                let threshold = (base_threshold * threshold_scaler).clamp(
                    summary.max_density * density_lowerbound_scaler,
                    summary.max_density * density_upperbound_scaler,
                );

                if density_map[(x, y)] > threshold {
                    *id = cluster_id;
                } else {
                    *id = -1;
                }
            } else {
                if density_map[(x, y)] > 0.0 {
                    *id = cluster_id;
                } else {
                    *id = -1;
                }
            }
        } else {
            *id = -1;
        }
    }

    let cluster_summaries: HashMap<i32, ClusterSummary> =
        graph.summary_map.iter().map(|(&id, &s)| (id, s)).collect();

    (cluster_map, cluster_summaries)
}
