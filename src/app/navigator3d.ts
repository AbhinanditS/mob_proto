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
// Latent-space slider — slerp between two endpoints, retrieve the nearest real
// image at each step. Like getNeighbors, the math lives entirely in .vec space
// (full 768-D, cosine via dot); the renderer separately reads the layout array.
// ---------------------------------------------------------------------------

// Spherical interpolation between two UNIT vectors, renormalized to be safe.
// The embeddings live on the unit sphere, so a plain lerp would cut across the
// chord and land off-sphere in incoherent dead space — slerp walks the geodesic
// arc instead. t=0 returns a, t=1 returns b (exactly, up to float precision).
export function slerp(a: Float32Array, b: Float32Array, t: number): Float32Array {
  const n = a.length;
  const out = new Float32Array(n);

  let d = dot(a, b);
  if (d > 1) d = 1;
  else if (d < -1) d = -1;

  const theta = Math.acos(d);
  const sinTheta = Math.sin(theta);

  // a≈b (sinTheta≈0) or antipodal a≈-b (d≈-1, arc is degenerate/undefined):
  // fall back to a normalized lerp rather than dividing by ~0.
  if (sinTheta < 1e-6) {
    for (let i = 0; i < n; i++) out[i] = a[i] * (1 - t) + b[i] * t;
  } else {
    const s0 = Math.sin((1 - t) * theta) / sinTheta;
    const s1 = Math.sin(t * theta) / sinTheta;
    for (let i = 0; i < n; i++) out[i] = a[i] * s0 + b[i] * s1;
  }

  // Renormalize back onto the unit sphere (guards float drift / lerp fallback).
  let norm = 0;
  for (let i = 0; i < n; i++) norm += out[i] * out[i];
  norm = Math.sqrt(norm) || 1;
  for (let i = 0; i < n; i++) out[i] /= norm;
  return out;
}

// Single nearest real image to queryVec by cosine over the full embedding.
// Unlike getNeighbors this KEEPS the top match. `exclude` skips given ids — used
// for the slider's intermediate points: on this dataset the two endpoints
// dominate every point on the arc, so excluding A and B is what lets a genuinely
// different real image surface in between. Returns null only for an empty set.
export function nearestImage(
  queryVec: Float32Array,
  items: NavItem[],
  exclude?: Set<string>
): Neighbor | null {
  let bestId = "";
  let bestScore = -Infinity;
  for (let i = 0; i < items.length; i++) {
    if (exclude && exclude.has(items[i].id)) continue;
    const s = dot(queryVec, items[i].vec);
    if (s > bestScore) {
      bestScore = s;
      bestId = items[i].id;
    }
  }
  return bestId ? { id: bestId, score: bestScore } : null;
}

// Top-k nearest real images to queryVec by cosine, ranked, keeping the top match
// (`exclude` skips given ids). Drives the live "neighbors of this arc point"
// strip while scrubbing. Distinct from getNeighbors, which drops the #1 hit.
export function topNeighbors(
  queryVec: Float32Array,
  items: NavItem[],
  k: number,
  exclude?: Set<string>
): Neighbor[] {
  const scored: Neighbor[] = [];
  for (let i = 0; i < items.length; i++) {
    if (exclude && exclude.has(items[i].id)) continue;
    scored.push({ id: items[i].id, score: dot(queryVec, items[i].vec) });
  }
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, k);
}

export interface PathStop {
  id: string;
  tStart: number; // the t at which this image first becomes the nearest
}

// Walk the geodesic from A to B in `steps` samples, recording the ordered set of
// DISTINCT real images the slider surfaces along the way. The endpoints are
// pinned (t=0 → A, t=1 → B); strictly-interior samples take the nearest image
// EXCLUDING A and B, so the path is the actual sequence of in-between discoveries
// rather than a bare A→B crossfade. Drives the 3D polyline and the filmstrip.
export function computePath(
  a: Float32Array,
  b: Float32Array,
  items: NavItem[],
  aId: string,
  bId: string,
  steps = 128
): PathStop[] {
  const exclude = new Set([aId, bId]);
  const stops: PathStop[] = [];
  let lastId = "";
  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    let id: string;
    if (i === 0) id = aId;
    else if (i === steps) id = bId;
    else {
      const nb = nearestImage(slerp(a, b, t), items, exclude);
      if (!nb) continue;
      id = nb.id;
    }
    if (id !== lastId) {
      stops.push({ id, tStart: t });
      lastId = id;
    }
  }
  return stops;
}
