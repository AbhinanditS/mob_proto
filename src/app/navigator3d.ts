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
    nNeighbors: Math.min(15, n - 1),
    minDist: 0.1,
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
