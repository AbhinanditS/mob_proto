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
// Latent-space slider — a coherent PATH from A to B across the real images.
//
// We deliberately do NOT sample the straight geodesic and snap to the nearest
// image: on a sparse set that line runs through empty space, so its "nearest"
// images relate to nothing (they're just the least-far point to an empty spot).
//
// Instead we route along the data's own minimum spanning tree (edges weighted by
// angular distance). The unique tree path between A and B is a minimum-bottleneck
// path: it provably minimizes the LARGEST hop, so there are no jarring jumps, and
// it threads through the dense manifold so consecutive images stay similar. This
// is far more robust than a greedy walk, which routinely dead-ends and has to
// jump straight to B. Like getNeighbors, the math is pure .vec space (full 768-D,
// cosine via dot); the renderer separately reads the layout array.
// ---------------------------------------------------------------------------

export interface PathStop {
  id: string;
  tStart: number; // slider position (0..1) where this image takes over
}

const ANGLE = (a: Float32Array, b: Float32Array) =>
  Math.acos(Math.max(-1, Math.min(1, dot(a, b))));

// Minimum spanning tree over all items (Prim, O(n²)), edges weighted by angular
// distance. Built once and cached for the session — it's independent of which
// endpoints are picked, like the UMAP layout. Returned as an adjacency list.
let mstCache: number[][] | null = null;
let mstCacheKey = "";
function buildMST(items: NavItem[]): number[][] {
  const n = items.length;
  const key = String(n);
  if (mstCache && mstCacheKey === key) return mstCache;

  const adj: number[][] = Array.from({ length: n }, () => []);
  if (n > 1) {
    const vec = items.map((it) => it.vec);
    const parent = new Int32Array(n).fill(-1);
    const inTree = new Uint8Array(n);
    const best = new Float64Array(n).fill(Infinity);
    best[0] = 0;
    for (let iter = 0; iter < n; iter++) {
      let u = -1;
      let bd = Infinity;
      for (let i = 0; i < n; i++) if (!inTree[i] && best[i] < bd) { bd = best[i]; u = i; }
      if (u < 0) break;
      inTree[u] = 1;
      for (let v = 0; v < n; v++) {
        if (inTree[v]) continue;
        const w = ANGLE(vec[u], vec[v]);
        if (w < best[v]) { best[v] = w; parent[v] = u; }
      }
    }
    for (let v = 0; v < n; v++) {
      if (parent[v] >= 0) { adj[v].push(parent[v]); adj[parent[v]].push(v); }
    }
  }
  mstCache = adj;
  mstCacheKey = key;
  return adj;
}

// The A→B path: the unique route through the MST, found by BFS over the tree.
// tStart is assigned by cumulative angular distance so the slider's spacing
// tracks the real distances between consecutive images.
export function buildWalk(
  items: NavItem[],
  aId: string,
  bId: string
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

  const adj = buildMST(items);
  const parent = new Int32Array(items.length).fill(-2);
  parent[A] = -1;
  const queue = [A];
  for (let h = 0; h < queue.length; h++) {
    const u = queue[h];
    if (u === B) break;
    for (const v of adj[u]) {
      if (parent[v] === -2) { parent[v] = u; queue.push(v); }
    }
  }

  const seq: number[] = [];
  for (let c = B; c >= 0; c = parent[c]) seq.unshift(c); // A … B

  const vec = items.map((it) => it.vec);
  const cum = [0];
  let total = 0;
  for (let i = 1; i < seq.length; i++) {
    total += ANGLE(vec[seq[i - 1]], vec[seq[i]]);
    cum.push(total);
  }
  return seq.map((s, i) => ({
    id: items[s].id,
    tStart: total > 0 ? cum[i] / total : i / Math.max(1, seq.length - 1),
  }));
}
