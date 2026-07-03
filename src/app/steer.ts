// Steer pad — pure logic (no React, no DOM).
//
// Turns a 2D pad position into a query vector for the EXISTING similarity
// pipeline: query = normalize(centroid + s·(x·axisX + y·axisY)), then rank by
// dot() like every other consumer. Axes are DIRECTIONS between two pole
// embeddings, applied around the image centroid — never absolute text points —
// so the SigLIP text/image modality gap cancels out.
//
// The axis functions take pole VECTORS, not words. Text-pole mode embeds two
// words and passes them in; a future image-pole mode passes two image .vecs
// instead and nothing else changes.
//
// Everything on the per-frame drag path (buildSteerQuery, scoreAll,
// scoresToRelevance) writes into caller-owned buffers — zero allocation per
// frame by design.

import { dot } from "./converge";

// s = STRENGTH_MULT × std of per-image projections onto the axis. ~2-3× moves
// the extremes clearly without collapsing the ranking onto one image.
export const STRENGTH_MULT = 2.5;
// Sharpens min-max-normalized relevance toward the top matches.
export const RELEVANCE_GAMMA = 2.0;
// Below this score spread the query is effectively the centroid — treat the
// field as neutral instead of min-max-amplifying noise into flicker.
export const MIN_SPREAD = 0.02;
// Spread at which relevance contrast reaches full strength.
export const SPREAD_REF = 0.08;
// Grid: lowest tile opacity while steering.
export const OPACITY_FLOOR = 0.25;
// Degenerate-axis guard (identical or near-parallel poles).
export const AXIS_EPS = 1e-3;

export interface SteerContext {
  n: number;
  dim: number;
  matrix: Float32Array; // n×dim packed row-major copy of item vecs
  centroid: Float32Array; // dim — mean of image embeddings, NOT normalized
}

// Pack the item vectors and their mean. Cheap (~1 ms at gallery scale); the
// caller memoizes on the items array identity, which uploads replace.
export function buildSteerContext(items: { vec: Float32Array }[]): SteerContext {
  const n = items.length;
  const dim = n > 0 ? items[0].vec.length : 0;
  const matrix = new Float32Array(n * dim);
  const centroid = new Float32Array(dim);
  for (let i = 0; i < n; i++) {
    const v = items[i].vec;
    matrix.set(v, i * dim);
    for (let d = 0; d < dim; d++) centroid[d] += v[d];
  }
  if (n > 0) for (let d = 0; d < dim; d++) centroid[d] /= n;
  return { n, dim, matrix, centroid };
}

function normalizeInto(v: Float32Array): boolean {
  let norm = 0;
  for (let i = 0; i < v.length; i++) norm += v[i] * v[i];
  norm = Math.sqrt(norm);
  if (norm < AXIS_EPS) return false;
  for (let i = 0; i < v.length; i++) v[i] /= norm;
  return true;
}

// Direction between two pole vectors: normalize(to − from).
// Null when the poles coincide (norm below AXIS_EPS).
export function deriveAxis(from: Float32Array, to: Float32Array): Float32Array | null {
  const out = new Float32Array(from.length);
  for (let i = 0; i < out.length; i++) out[i] = to[i] - from[i];
  return normalizeInto(out) ? out : null;
}

// Gram-Schmidt: normalize(y − dot(y,x)·x). Null when y is near-parallel to x
// (the residual would be noise, not a direction).
export function orthogonalize(y: Float32Array, x: Float32Array): Float32Array | null {
  const p = dot(y, x);
  const out = new Float32Array(y.length);
  for (let i = 0; i < out.length; i++) out[i] = y[i] - p * x[i];
  return normalizeInto(out) ? out : null;
}

// Per-axis strength: STRENGTH_MULT × std of the images' projections onto the
// axis. Derivation-time only (O(n·d)), never on the drag path.
export function axisStrength(ctx: SteerContext, axis: Float32Array): number {
  const { n, dim, matrix } = ctx;
  if (n === 0) return 0;
  let sum = 0;
  let sumSq = 0;
  for (let i = 0; i < n; i++) {
    let p = 0;
    const base = i * dim;
    for (let d = 0; d < dim; d++) p += matrix[base + d] * axis[d];
    sum += p;
    sumSq += p * p;
  }
  const mean = sum / n;
  const variance = Math.max(0, sumSq / n - mean * mean);
  return STRENGTH_MULT * Math.sqrt(variance);
}

// out = normalize(centroid + sx·x·axisX + sy·y·axisY), in place.
// A null axis contributes nothing (one-axis steering just works).
export function buildSteerQuery(
  ctx: SteerContext,
  axisX: Float32Array | null,
  axisY: Float32Array | null,
  x: number,
  y: number,
  sx: number,
  sy: number,
  out: Float32Array
): Float32Array {
  const { dim, centroid } = ctx;
  const wx = axisX ? sx * x : 0;
  const wy = axisY ? sy * y : 0;
  for (let d = 0; d < dim; d++) {
    let v = centroid[d];
    if (axisX) v += wx * axisX[d];
    if (axisY) v += wy * axisY[d];
    out[d] = v;
  }
  normalizeInto(out);
  return out;
}

// Cosine of every image against the query (vectors are pre-normalized, so a
// bare dot suffices — same contract as the rest of the app).
export function scoreAll(query: Float32Array, ctx: SteerContext, out: Float32Array): void {
  const { n, dim, matrix } = ctx;
  for (let i = 0; i < n; i++) {
    let s = 0;
    const base = i * dim;
    for (let d = 0; d < dim; d++) s += matrix[base + d] * query[d];
    out[i] = s;
  }
}

// Map raw scores to display relevance in [0,1]; returns the argmax index.
// Contrast is damped by the score spread so a near-centroid query (pad center)
// reads as a neutral field instead of noise stretched to full range. This is
// the single canonical mapping — grid and 3D both consume it, so they always
// agree on what "relevant" looks like.
export function scoresToRelevance(scores: Float32Array, n: number, out: Float32Array): number {
  if (n === 0) return -1;
  let min = scores[0];
  let max = scores[0];
  let top = 0;
  for (let i = 1; i < n; i++) {
    const s = scores[i];
    if (s < min) min = s;
    if (s > max) {
      max = s;
      top = i;
    }
  }
  const spread = max - min;
  if (spread < MIN_SPREAD) {
    out.fill(1, 0, n);
    return top;
  }
  const t = Math.min(1, (spread - MIN_SPREAD) / SPREAD_REF);
  for (let i = 0; i < n; i++) {
    const norm = (scores[i] - min) / spread;
    out[i] = 1 - t * (1 - Math.pow(norm, RELEVANCE_GAMMA));
  }
  return top;
}
