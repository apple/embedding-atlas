/* tslint:disable */
/* eslint-disable */
/**
* @param {DensityMap} input
* @param {FindClustersOptions} options
* @returns {FindClustersResult}
*/
export function find_clusters(input: DensityMap, options: FindClustersOptions): FindClustersResult;

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


/**
*/
export class DensityMap {
  free(): void;
/**
* @param {number} width
* @param {number} height
* @param {Float32Array} data
*/
  constructor(width: number, height: number, data: Float32Array);
/**
* @returns {number}
*/
  width(): number;
/**
* @returns {number}
*/
  height(): number;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly __wbg_densitymap_free: (a: number) => void;
  readonly densitymap_new: (a: number, b: number, c: number, d: number) => number;
  readonly densitymap_width: (a: number) => number;
  readonly densitymap_height: (a: number) => number;
  readonly find_clusters: (a: number, b: number) => number;
  readonly __wbindgen_malloc: (a: number, b: number) => number;
  readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;
/**
* Instantiates the given `module`, which can either be bytes or
* a precompiled `WebAssembly.Module`.
*
* @param {SyncInitInput} module
*
* @returns {InitOutput}
*/
export function initSync(module: SyncInitInput): InitOutput;

/**
* If `module_or_path` is {RequestInfo} or {URL}, makes a request and
* for everything else, calls `WebAssembly.instantiate` directly.
*
* @param {InitInput | Promise<InitInput>} module_or_path
*
* @returns {Promise<InitOutput>}
*/
export default function __wbg_init (module_or_path?: InitInput | Promise<InitInput>): Promise<InitOutput>;
