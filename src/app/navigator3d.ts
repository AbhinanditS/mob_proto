// 3D latent-space navigator — pure logic (no React, no three.js).
//
// Two jobs, deliberately kept apart:
//   1. computeLayout() — UMAP 768-D → 3-D, run ONCE and cached for the session.
//      The returned positions are for RENDERING ONLY.
//   2. getNeighbors() — closest matches by cosine similarity on the FULL whitened
//      SigLIP vectors, reusing the existing dot(). It never reads the layout.
//
// The one hard rule: neighbor logic touches .vec via dot(); the renderer touches
// the layout array. If neighbor code ever indexes the layout, that is the bug.

import { UMAP } from "umap-js";
import { dot } from "./converge";

export interface NavItem {
  id: string;
  vec: Float32Array;
}

// World radius the cached cloud is normalized to fill. The scene frames the
// camera and sizes sprites relative to this.
export const LAYOUT_RADIUS = 50;

// Deterministic RNG so the cached layout is stable across reopens/reloads — the
// idle cloud is the hero shot, so its framing shouldn't reshuffle each session.
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

let layoutCache: Float32Array | null = null;
let layoutCacheKey = "";

// UMAP 768 → 3, computed once and cached at module scope for the session.
// Returns a flat Float32Array (n*3), centered on the origin and scaled so the
// farthest point sits at LAYOUT_RADIUS. RENDERING ONLY — never read for similarity.
export function computeLayout(items: NavItem[]): Float32Array {
  const key = String(items.length);
  if (layoutCache && layoutCacheKey === key) return layoutCache;

  const n = items.length;
  const out = new Float32Array(n * 3);
  if (n <= 1) {
    layoutCache = out; // 0 or 1 points → origin
    layoutCacheKey = key;
    return out;
  }

  const X = items.map((it) => Array.from(it.vec));
  const umap = new UMAP({
    nComponents: 3,
    // 15-50 is the standard UMAP guidance range; 30 weighs more toward
    // preserving global structure than the previous low-end value.
    nNeighbors: Math.min(30, n - 1),
    minDist: 0.1,
    // Explicit rather than relying on umap-js's implicit size-based default —
    // pins the convergence budget so behavior doesn't shift across versions.
    nEpochs: 500,
    random: mulberry32(1234567),
  });
  const embedding = umap.fit(X); // number[][], n × 3

  // Center on the centroid.
  let cx = 0, cy = 0, cz = 0;
  for (const p of embedding) {
    cx += p[0]; cy += p[1]; cz += p[2];
  }
  cx /= n; cy /= n; cz /= n;

  // Scale so the farthest point lands at LAYOUT_RADIUS.
  let maxR = 1e-6;
  for (const p of embedding) {
    const dx = p[0] - cx, dy = p[1] - cy, dz = p[2] - cz;
    const r = Math.sqrt(dx * dx + dy * dy + dz * dz);
    if (r > maxR) maxR = r;
  }
  const s = LAYOUT_RADIUS / maxR;
  for (let i = 0; i < n; i++) {
    out[i * 3 + 0] = (embedding[i][0] - cx) * s;
    out[i * 3 + 1] = (embedding[i][1] - cy) * s;
    out[i * 3 + 2] = (embedding[i][2] - cz) * s;
  }

  layoutCache = out;
  layoutCacheKey = key;
  return out;
}

export interface Neighbor {
  id: string;
  score: number;
}

// Closest matches to queryVec by cosine similarity on the FULL whitened SigLIP
// vectors. Reuses dot() (vectors are L2-normalized, so dot == cosine). This is
// the ONLY source of neighbors; it never touches the layout array.
export function getNeighbors(
  queryVec: Float32Array,
  items: NavItem[],
  k: number
): Neighbor[] {
  const scored: Neighbor[] = new Array(items.length);
  for (let i = 0; i < items.length; i++) {
    scored[i] = { id: items[i].id, score: dot(queryVec, items[i].vec) };
  }
  scored.sort((a, b) => b.score - a.score);
  // Drop the query itself (score ≈ 1 at the top) and return the next k.
  return scored.slice(1, k + 1);
}

// ---------------------------------------------------------------------------
// Probe mode — k-NN and density purely over the 3-D layout array (Euclidean
// distance), never over .vec. This answers "what's near this point in the
// RENDERED cloud," which is a distinct question from getNeighbors() above
// ("what's near this vector in the full embedding space"). Mixing the two
// would defeat the point: the layout has no inverse, so probing must stay in
// layout space to make sense of arbitrary (non-image) query points.
// ---------------------------------------------------------------------------

export interface PositionNeighbor {
  index: number;
  dist: number; // Euclidean distance in layout (render) space
}

// Closest points to `point` by Euclidean distance in the 3-D layout array.
// Brute-force O(n) distance + sort; no caching (the query point changes every
// call). Safe at gallery scale on a debounce — see Navigator3D's probe wiring.
export function kNearestByPosition(
  positions: Float32Array,
  n: number,
  point: { x: number; y: number; z: number },
  k: number
): PositionNeighbor[] {
  const scored: PositionNeighbor[] = new Array(n);
  for (let i = 0; i < n; i++) {
    const dx = positions[i * 3] - point.x;
    const dy = positions[i * 3 + 1] - point.y;
    const dz = positions[i * 3 + 2] - point.z;
    scored[i] = { index: i, dist: Math.sqrt(dx * dx + dy * dy + dz * dz) };
  }
  scored.sort((a, b) => a.dist - b.dist);
  return scored.slice(0, k);
}

// Per-point local-density scalar: mean distance to each point's k nearest
// neighbors IN LAYOUT SPACE, normalized/inverted to [0,1] (1 = densest region,
// 0 = sparsest). Computed once per session (O(n² log n)) and cached at module
// scope like layoutCache — re-entering probe mode after the first time is free.
let densityCache: Float32Array | null = null;
let densityCacheKey = "";

export function computeDensity(positions: Float32Array, n: number, k = 6): Float32Array {
  const key = `${n}:${k}`;
  if (densityCache && densityCacheKey === key) return densityCache;

  const raw = new Float32Array(n);
  const dists = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (j === i) { dists[j] = Infinity; continue; }
      const dx = positions[i * 3] - positions[j * 3];
      const dy = positions[i * 3 + 1] - positions[j * 3 + 1];
      const dz = positions[i * 3 + 2] - positions[j * 3 + 2];
      dists[j] = Math.sqrt(dx * dx + dy * dy + dz * dz);
    }
    const sorted = Array.from(dists).sort((a, b) => a - b);
    let sum = 0;
    const kk = Math.min(k, n - 1);
    for (let t = 0; t < kk; t++) sum += sorted[t];
    raw[i] = kk > 0 ? sum / kk : 0;
  }

  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < n; i++) {
    if (raw[i] < lo) lo = raw[i];
    if (raw[i] > hi) hi = raw[i];
  }
  const out = new Float32Array(n);
  const span = Math.max(1e-6, hi - lo);
  for (let i = 0; i < n; i++) out[i] = 1 - (raw[i] - lo) / span;

  densityCache = out;
  densityCacheKey = key;
  return out;
}
