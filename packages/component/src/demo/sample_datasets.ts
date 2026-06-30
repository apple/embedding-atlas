// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

export interface DatasetConfig {
  numPoints: number;
  numCategories: number;
  numSubClusters: number;
}

export interface Row {
  identifier: number;
  x: number;
  y: number;
  category: number;
  text: string;
}

export function generateSampleDataset(config: DatasetConfig): Row[] {
  let rng = RNG(42);
  let clusters: {
    x: number;
    y: number;
    sigmaX: number;
    sigmaY: number;
    angle: number;
    category: number;
    text: string;
  }[] = [];
  for (let c = 0; c < config.numCategories; c++) {
    for (let j = 0; j < config.numSubClusters; j++) {
      clusters.push({
        x: randomUniform(-10, 10, rng),
        y: randomUniform(-10, 10, rng),
        sigmaX: randomUniform(0.2, 2, rng),
        sigmaY: randomUniform(0.2, 2, rng),
        angle: rng() * Math.PI * 2,
        category: c,
        text: randomChoices(words, 3, rng).join(", "),
      });
    }
  }
  let result: Row[] = [];
  for (let i = 0; i < config.numPoints; i++) {
    let cluster = clusters[Math.floor(rng() * clusters.length)];
    let x = randomNormal(cluster.x, cluster.sigmaX, rng);
    let y = randomNormal(cluster.y, cluster.sigmaY, rng);
    result.push({
      identifier: i,
      x: Math.cos(cluster.angle) * x - Math.sin(cluster.angle) * y,
      y: Math.sin(cluster.angle) * x + Math.cos(cluster.angle) * y,
      category: cluster.category,
      text: cluster.text + ` (${i})`,
    });
  }
  return result;
}

export interface Row3D {
  identifier: number;
  x: number;
  y: number;
  z: number;
  category: number;
  text: string;
}

/**
 * Generates a deterministic 3D dataset of well-separated Gaussian blobs (one per
 * category). The blobs are placed apart in all three axes so that rotating the
 * camera clearly reveals depth structure.
 */
export function generateSampleDataset3D(config: { numPoints: number; numCategories: number }): Row3D[] {
  let rng = RNG(42);
  let centers: { x: number; y: number; z: number; sigma: number; category: number; text: string }[] = [];
  for (let c = 0; c < config.numCategories; c++) {
    centers.push({
      x: randomUniform(-6, 6, rng),
      y: randomUniform(-6, 6, rng),
      z: randomUniform(-6, 6, rng),
      sigma: randomUniform(1.0, 2.5, rng),
      category: c,
      text: randomChoices(words, 2, rng).join(", "),
    });
  }
  let result: Row3D[] = [];
  for (let i = 0; i < config.numPoints; i++) {
    let cl = centers[i % config.numCategories];
    result.push({
      identifier: i,
      x: randomNormal(cl.x, cl.sigma, rng),
      y: randomNormal(cl.y, cl.sigma, rng),
      z: randomNormal(cl.z, cl.sigma, rng),
      category: cl.category,
      text: cl.text + ` (#${i})`,
    });
  }
  return result;
}

class SplitMix64 {
  seed: bigint;

  constructor(seed: number) {
    this.seed = BigInt(seed);
  }

  next() {
    let z = (this.seed += BigInt("0x9E3779B97F4A7C15"));
    z = (z ^ (z >> BigInt(30))) * BigInt("0xBF58476D1CE4E5B9");
    z = (z ^ (z >> BigInt(27))) * BigInt("0x94D049BB133111EB");
    return z ^ (z >> BigInt(31));
  }

  nextNumber() {
    return Number(this.next() & BigInt("0xFFFFFFFFFFFFFFFF"));
  }
}

function RNG(seed: number) {
  let algo = new SplitMix64(seed);
  return () => {
    return algo.nextNumber() / 0xffffffffffffffff;
  };
}

function randomUniform(a: number, b: number, rng: () => number) {
  return a + (b - a) * rng();
}

function randomNormal(mean: number, stdev: number, rng: () => number) {
  const u = 1 - rng();
  const v = rng();
  const z = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  return z * stdev + mean;
}

function randomChoices(list: string[], count: number, rng: () => number) {
  let result: string[] = [];
  for (let i = 0; i < count; i++) {
    result.push(list[Math.floor(rng() * list.length)]);
  }
  return result;
}

let words = [
  "Gala",
  "Fuji",
  "Granny Smith",
  "Honeycrisp",
  "Red Delicious",
  "Golden Delicious",
  "Braeburn",
  "Pink Lady",
  "Cortland",
  "Jonagold",
  "Empire",
  "Ambrosia",
  "Jazz",
  "Cameo",
  "Envy",
  "Mutsu",
  "Opal",
  "Winesap",
];
