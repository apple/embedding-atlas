// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { stemmer } from "stemmer";

import type { Rectangle } from "../utils.js";
import { stopWords as defaultStopWords } from "./stop_words.js";

/** A text summarizer based on c-TF-IDF (https://arxiv.org/pdf/2203.05794) */
export class TFIDFSummarizer {
  private segmenter: Intl.Segmenter;
  private binning: XYBinning;
  private stopWords: Set<string>;
  private key2RegionIndices: Map<number, number[]>;
  private frequencyPerClass: Map<string, number>[];
  private frequencyAll: Map<string, number>;

  /** Create a new TFIDFSummarizer */
  constructor(options: {
    binning: { xMin: number; xStep: number; yMin: number; yStep: number };
    regions: Rectangle[][];
    stopWords?: string[];
  }) {
    this.binning = new XYBinning(
      options.binning.xMin,
      options.binning.yMin,
      options.binning.xStep,
      options.binning.yStep,
    );
    this.segmenter = new Intl.Segmenter(undefined, { granularity: "word" });
    this.stopWords = new Set(options.stopWords ?? defaultStopWords);

    this.frequencyPerClass = options.regions.map(() => new Map());
    this.frequencyAll = new Map();

    // Generate key2RegionIndices, a map from xy key to region index
    this.key2RegionIndices = new Map();
    for (let i = 0; i < options.regions.length; i++) {
      let keys = this.binning.keys(options.regions[i]);
      for (let k of keys) {
        let v = this.key2RegionIndices.get(k);
        if (v != null) {
          v.push(i);
        } else {
          this.key2RegionIndices.set(k, [i]);
        }
      }
    }
  }

  /** Add data to the summarizer */
  add(data: { x: ArrayLike<number>; y: ArrayLike<number>; text: ArrayLike<string> }) {
    for (let i = 0; i < data.text.length; i++) {
      let key = this.binning.key(data.x[i], data.y[i]);
      let indices = this.key2RegionIndices.get(key);
      if (indices == null) {
        continue;
      }
      for (let s of this.segmenter.segment(data.text[i])) {
        let word = s.segment.toLowerCase().trim();
        if (word.length > 1) {
          for (let idx of indices) {
            incrementMap(this.frequencyPerClass[idx], word);
          }
          incrementMap(this.frequencyAll, word);
        }
      }
    }
  }

  isStopWord(word: string) {
    // Consider words in the stop words list or pure numbers as stop words.
    return this.stopWords.has(word) || /^[0-9]+$/.test(word);
  }

  summarize(limit: number = 4): string[][] {
    // Aggregate the frequencies by stemmed words
    let frequencyAllStem = aggregateByStem(this.frequencyAll);
    let frequencyPerClassStem = this.frequencyPerClass.map(aggregateByStem);

    // Average number of words per class
    let averageWords =
      frequencyPerClassStem.map((x) => x.values().reduce((a, b) => a + b[1], 0)).reduce((a, b) => a + b, 0) /
      frequencyPerClassStem.length;

    return frequencyPerClassStem.map((wordMap) => {
      // Compute TF-IDF
      let entries = Array.from(
        wordMap.entries().map(([key, [word, tf]]) => {
          let df = frequencyAllStem.get(key)?.[1] ?? 1;
          let idf = Math.log(1 + averageWords / df);
          return {
            word: word,
            tf: tf,
            df: df,
            idf: idf,
            tfIDF: tf * idf,
          };
        }),
      );
      entries = entries.filter((x) => !this.isStopWord(x.word) && x.df >= 2);
      entries = entries.sort((a, b) => b.tfIDF - a.tfIDF);
      return entries.slice(0, limit).map((x) => x.word);
    });
  }
}

class XYBinning {
  private xMin: number;
  private yMin: number;
  private xStep: number;
  private yStep: number;

  constructor(xMin: number, yMin: number, xStep: number, yStep: number) {
    this.xMin = xMin;
    this.yMin = yMin;
    this.xStep = xStep;
    this.yStep = yStep;
  }

  key(x: number, y: number) {
    let ix = Math.floor((x - this.xMin) / this.xStep);
    let iy = Math.floor((y - this.yMin) / this.yStep);
    return ix + iy * 32768;
  }

  keys(rects: Rectangle[]): number[] {
    let keys = new Set<number>();
    for (let { xMin, yMin, xMax, yMax } of rects) {
      let xiLowerBound = Math.floor((xMin - this.xMin) / this.xStep);
      let xiUpperBound = Math.floor((xMax - this.xMin) / this.xStep);
      let yiLowerBound = Math.floor((yMin - this.yMin) / this.yStep);
      let yiUpperBound = Math.floor((yMax - this.yMin) / this.yStep);
      for (let xi = xiLowerBound; xi <= xiUpperBound; xi++) {
        for (let yi = yiLowerBound; yi <= yiUpperBound; yi++) {
          let p = yi * 32768 + xi;
          keys.add(p);
        }
      }
    }
    return Array.from(keys);
  }
}

function incrementMap<K>(map: Map<K, number>, key: K) {
  let c = map.get(key) ?? 0;
  map.set(key, c + 1);
}

/** Aggregate words by their stems and track the most frequent version.
 * Returns a map with stemmed words as keys, and the most frequent version and total count as values. */
function aggregateByStem(inputMap: Map<string, number>): Map<string, [string, number]> {
  const result = new Map();
  for (const [word, count] of inputMap.entries()) {
    const s = stemmer(word);
    if (result.has(s)) {
      const value = result.get(s);
      value[1] += count;
      if ((inputMap.get(value[0]) ?? 0) < count) {
        value[0] = word;
      }
    } else {
      result.set(s, [word, count]);
    }
  }
  return result;
}
