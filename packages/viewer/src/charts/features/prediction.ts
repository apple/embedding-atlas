/**
 * Per-class direction score for multi-class prediction. One entry per
 * (non-empty) class, ordered to match the `classNames` array. Built from the
 * one-vs-rest 2×2 contingency for that class. Auxiliary fields are for
 * tooltip/display only — ranking uses the feature-level `strength`.
 */
export interface PerClassScore {
  className: string;
  /**
   * Signed, confidence-aware direction score: `sign(LOR)·max(0, |LOR| − z·SE)`
   * on the Haldane–Anscombe-corrected one-vs-rest table. > 0 = feature
   * confidently shifts mass toward this class; < 0 = away; 0 = no confident
   * direction (CI on the LOR crosses zero).
   */
  score: number;
  lor: number;
  lorCiLow: number;
  lorCiHigh: number;
  oddsRatio: number;
  /** P(class | feature present) = a / m. */
  pClassGivenFeature: number;
  /** P(class) = N_class / N. */
  pClass: number;
  /** Smoothed log-lift: log( ((a+1)/(m+K)) / P(class) ). */
  logLift: number;
  /** a — rows where the feature is present and class is this one. */
  supportInClass: number;
}

export interface Predict {
  /**
   * Magnitude of the predictive signal, used for sorting / bar size. Always >= 0.
   *
   * - Binary: `max(0, |LOR| − z·SE)` — Bonferroni-adjusted lower bound on the
   *   absolute log-odds ratio at family-wise 95% confidence across all
   *   features. Zero when the CI crosses zero (no confident direction).
   * - Multi-class: small-sample-debiased mutual information in nats,
   *   `max(0, MI − (K−1)/(2N))`, between feature presence and the class column.
   */
  strength: number;
  /**
   * For binary classification only (classNames.length === 2): the index in
   * classNames of the class the feature predicts toward (sign of LOR).
   * Set even when strength === 0. Omitted for multi-class.
   */
  direction?: 0 | 1;
  /**
   * For binary classification only: signed φ coefficient on the
   * Haldane–Anscombe-corrected 2×2 contingency, in [−1, 1]. For tooltip /
   * display; ranking still uses `strength`. Omitted for multi-class.
   */
  phi?: number;
  /**
   * Multi-class only: `m`, total rows where the feature is present (the
   * support behind the MI / per-class scores). Omitted for binary.
   */
  support?: number;
  /**
   * Multi-class only: signed per-class direction scores, ordered to match
   * `classNames`. Omitted for binary.
   */
  perClass?: PerClassScore[];
}

/**
 * Haldane–Anscombe-corrected log-odds ratio and its signed, confidence-aware
 * lower bound for a 2×2 contingency table with raw cells:
 *
 *     | class=i  class≠i
 *   f |   a        b
 *  ¬f |   c        d
 *
 * Adds 0.5 to every cell (Haldane–Anscombe) so the LOR and Woolf SE stay finite
 * when a cell is zero, then returns:
 *
 *   lor   = log((a'·d')/(b'·c'))
 *   se    = sqrt(1/a' + 1/b' + 1/c' + 1/d')          (Woolf)
 *   score = sign(LOR)·max(0, |LOR| − z·SE)           — 0 when the z·SE
 *           confidence interval on the LOR crosses zero (no confident direction)
 *
 * Shared by the binary and multi-class (one-vs-rest) paths.
 */
function signedLorLowerBound(
  a: number,
  b: number,
  c: number,
  d: number,
  z: number,
): { lor: number; se: number; score: number } {
  const ah = a + 0.5;
  const bh = b + 0.5;
  const ch = c + 0.5;
  const dh = d + 0.5;
  const lor = Math.log((ah * dh) / (bh * ch));
  const se = Math.sqrt(1 / ah + 1 / bh + 1 / ch + 1 / dh);
  const score = Math.sign(lor) * Math.max(0, Math.abs(lor) - z * se);
  return { lor, se, score };
}

/**
 * Predict info for binary classification.
 *
 * Builds the 2×2 contingency [{feature present, absent} × {classNames[1] = positive,
 * classNames[0] = negative}], applies the Haldane–Anscombe (+0.5) correction,
 * and ranks features by the Bonferroni-adjusted lower bound on |log-odds-ratio|:
 *
 *   strength  = max(0, |LOR| − z·SE)         (Woolf SE)
 *   direction = 1 iff LOR > 0                (predicts classNames[1])
 *   phi       = signed φ coefficient on the corrected counts (for display)
 *
 * `z` is the family-wise critical value passed in by the caller (precomputed
 * from the total feature count F via `Φ⁻¹(1 − 0.025 / F)`). Returns undefined
 * when either class has zero rows under the current selection (no contrast).
 */
export function buildPredictBinary(
  nByClass: Map<string, number>,
  classNames: string[],
  classTotals: Map<string, number>,
  z: number,
): Predict | undefined {
  const aRaw = nByClass.get(classNames[1]) ?? 0;
  const bRaw = nByClass.get(classNames[0]) ?? 0;
  const N1 = classTotals.get(classNames[1]) ?? 0;
  const N0 = classTotals.get(classNames[0]) ?? 0;

  // Both classes must have rows under the current selection. A cross-filter
  // can empty one class (classTotals here are the *selected* totals), leaving
  // no contrast to predict from; without this guard the Haldane–Anscombe
  // correction would still yield a spurious non-zero LOR for a class with zero
  // selected rows. Mirrors the multi-class path's K < 2 early-out.
  if (N1 <= 0 || N0 <= 0) {
    return undefined;
  }

  // Defensive clamp: a should never exceed its column total, but rounding
  // or stale globals could in theory push c/d negative.
  const cRaw = Math.max(0, N1 - aRaw);
  const dRaw = Math.max(0, N0 - bRaw);

  const { lor, score } = signedLorLowerBound(aRaw, bRaw, cRaw, dRaw, z);
  const strength = Math.abs(score); // |signed lower bound| = max(0, |LOR| − z·SE)
  const direction: 0 | 1 = lor > 0 ? 1 : 0;

  // φ coefficient on the Haldane–Anscombe-corrected cells (display only).
  const a = aRaw + 0.5;
  const b = bRaw + 0.5;
  const c = cRaw + 0.5;
  const d = dRaw + 0.5;
  const phiNum = a * d - b * c;
  const phiDenSq = (a + b) * (c + d) * (a + c) * (b + d);
  const phi = phiDenSq > 0 ? phiNum / Math.sqrt(phiDenSq) : 0;

  return { strength, direction, phi };
}

/**
 * Predict info for multi-class classification.
 *
 * Computes two quantities on the 2×K contingency of {feature present/absent} ×
 * {class}, both of which penalize rare features so small-sample noise doesn't
 * dominate the ranking:
 *
 *   strength  — small-sample-debiased mutual information (nats),
 *               `max(0, MI − (K−1)/(2N))`. The headline sortable score.
 *   perClass  — signed one-vs-rest LOR lower bounds (Haldane–Anscombe
 *               corrected), one per class, giving direction + confidence.
 *
 * `classNames` is the ordered class list; classes with zero rows under the
 * current selection are dropped (K = number of non-empty classes). `z` is the
 * per-class confidence critical value precomputed by the caller. Returns
 * undefined for constant features (m == 0 or m == N) or when fewer than two
 * non-empty classes remain.
 */
export function buildPredictMulticlass(
  nByClass: Map<string, number>,
  classNames: string[],
  classTotals: Map<string, number>,
  total: number,
  z: number,
): Predict | undefined {
  const N = total;
  if (N <= 0) {
    return undefined;
  }

  // Drop empty classes up front; K is the number of non-empty classes.
  const classes = classNames.filter((c) => (classTotals.get(c) ?? 0) > 0);
  const K = classes.length;
  if (K < 2) {
    return undefined;
  }

  const a = classes.map((c) => nByClass.get(c) ?? 0);
  const nClass = classes.map((c) => classTotals.get(c) ?? 0);
  const m = a.reduce((s, x) => s + x, 0);
  // Constant feature: present nowhere or everywhere — no signal.
  if (m <= 0 || m >= N) {
    return undefined;
  }

  // Step 1 — debiased mutual information on the full 2×K table (nats).
  const pF = m / N;
  const pNotF = (N - m) / N;
  let mi = 0;
  for (let i = 0; i < K; i++) {
    const pC = nClass[i] / N;
    // feature present, class i
    if (a[i] > 0) {
      const pJoint = a[i] / N;
      mi += pJoint * Math.log(pJoint / (pF * pC));
    }
    // feature absent, class i
    const aAbsent = nClass[i] - a[i];
    if (aAbsent > 0) {
      const pJoint = aAbsent / N;
      mi += pJoint * Math.log(pJoint / (pNotF * pC));
    }
  }
  const strength = Math.max(0, mi - (K - 1) / (2 * N));

  // Step 2 + 3 — per-class one-vs-rest LOR lower bounds and display aux values.
  const perClass: PerClassScore[] = [];
  for (let i = 0; i < K; i++) {
    const A = a[i]; // feature present, class i
    const B = m - a[i]; // feature present, class ≠ i
    const C = Math.max(0, nClass[i] - a[i]); // feature absent, class i
    const D = Math.max(0, N - m - (nClass[i] - a[i])); // feature absent, class ≠ i

    const { lor, se, score } = signedLorLowerBound(A, B, C, D, z);

    const pClass = nClass[i] / N;
    perClass.push({
      className: classes[i],
      score,
      lor,
      lorCiLow: lor - z * se,
      lorCiHigh: lor + z * se,
      oddsRatio: Math.exp(lor),
      pClassGivenFeature: A / m,
      pClass,
      logLift: Math.log((A + 1) / (m + K) / pClass),
      supportInClass: A,
    });
  }

  return { strength, support: m, perClass };
}
