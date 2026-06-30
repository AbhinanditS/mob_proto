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

// ---------------------------------------------------------------------------
// Latent-space slider — a coherent WALK from A to B across the real images.
//
// We deliberately do NOT sample the straight geodesic and snap to the nearest
// image: on a sparse set that line runs through empty space, so its "nearest"
// images relate to nothing (they're just the least-far point to an empty spot).
// Instead each step hops to a near-neighbor of the CURRENT image — so every
// image shown is genuinely similar to the one before it — drifting toward B.
// The chain always reaches B. Like getNeighbors, the math is pure .vec space
// (full 768-D, cosine via dot); the renderer separately reads the layout array.
// ---------------------------------------------------------------------------

export interface PathStop {
  id: string;
  tStart: number; // slider position (0..1) where this image takes over
}

// Build a connected A→B chain by repeated small neighbor hops. From the current
// image we look at its `k` nearest neighbors and, among those that move closer
// to B than we are now, take the SMALLEST hop (the candidate most similar to the
// current image). That keeps each transition tight while still making progress,
// yielding a smooth walk rather than a couple of big jumps. tStart is assigned by
// cumulative angular distance so the slider's spacing tracks real distances.
export function buildWalk(
  items: NavItem[],
  aId: string,
  bId: string,
  k = 12
): PathStop[] {
  const idx = new Map(items.map((it, i) => [it.id, i]));
  const A = idx.get(aId);
  const B = idx.get(bId);
  if (A === undefined || B === undefined || A === B) {
    return [
      { id: aId, tStart: 0 },
      { id: bId, tStart: 1 },
    ];
  }

  const vec = items.map((it) => it.vec);
  const cos = (i: number, j: number) => dot(vec[i], vec[j]);
  const cosB = vec.map((v) => dot(v, vec[B])); // similarity of every image to B

  const seq: number[] = [A];
  const visited = new Set<number>([A]);
  let cur = A;

  for (let guard = 0; cur !== B && guard < 200; guard++) {
    // The k nearest unvisited neighbors of the current image.
    const scored: Array<[number, number]> = [];
    for (let j = 0; j < items.length; j++) {
      if (j === cur || visited.has(j)) continue;
      scored.push([j, cos(cur, j)]);
    }
    scored.sort((x, y) => y[1] - x[1]);
    const cand = scored.slice(0, k).map((s) => s[0]);

    // B in reach → finish on it.
    if (cand.includes(B)) {
      seq.push(B);
      break;
    }
    // Smallest hop that still moves us closer to B.
    let best = -1;
    let bestStep = -Infinity;
    for (const j of cand) {
      if (cosB[j] > cosB[cur] && cos(cur, j) > bestStep) {
        bestStep = cos(cur, j);
        best = j;
      }
    }
    // No progressing neighbor (local maximum) → jump straight to B and stop.
    if (best < 0) {
      seq.push(B);
      break;
    }
    seq.push(best);
    visited.add(best);
    cur = best;
  }
  if (seq[seq.length - 1] !== B) seq.push(B);

  // Distribute slider positions by cumulative angular distance along the chain.
  const cum = [0];
  let total = 0;
  for (let i = 1; i < seq.length; i++) {
    total += Math.acos(Math.max(-1, Math.min(1, cos(seq[i - 1], seq[i]))));
    cum.push(total);
  }
  return seq.map((s, i) => ({
    id: items[s].id,
    tStart: total > 0 ? cum[i] / total : i / Math.max(1, seq.length - 1),
  }));
}
