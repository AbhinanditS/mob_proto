// Pure math for the offline pipeline: duplicates, hierarchical clusters, the
// 3D layout and zero-shot naming. No I/O and no models; scripts/precompute.ts
// feeds it vectors and writes the result.

import { UMAP } from "umap-js";
import { dot } from "../src/lib/vectors";

export type Vec = Float32Array;

export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function meanVector(vs: Vec[]): Vec {
  const m = new Float32Array(vs[0].length);
  for (const v of vs) for (let i = 0; i < m.length; i++) m[i] += v[i];
  for (let i = 0; i < m.length; i++) m[i] /= vs.length;
  return m;
}

export function similarityMatrix(vs: Vec[]): Float64Array {
  const n = vs.length;
  const S = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    S[i * n + i] = 1;
    for (let j = i + 1; j < n; j++) S[i * n + j] = S[j * n + i] = dot(vs[i], vs[j]);
  }
  return S;
}

// Exact duplicates (the same pin saved twice). Returns indices to drop; the
// first occurrence of each image is kept.
export function duplicateIndices(a: Vec[], b: Vec[], thresholdA = 0.995, thresholdB = 0.99): Set<number> {
  const drop = new Set<number>();
  for (let i = 0; i < a.length; i++) {
    for (let j = 0; j < i; j++) {
      if (drop.has(j)) continue;
      if (dot(a[i], a[j]) >= thresholdA && dot(b[i], b[j]) >= thresholdB) {
        drop.add(i);
        break;
      }
    }
  }
  return drop;
}

// Mean similarity to the k nearest others. Used twice: as the CSLS hub
// penalty, and to tell well-connected images from loose ones.
export function localDensity(S: Float64Array, n: number, k: number): Float64Array {
  const r = new Float64Array(n);
  const row = new Float64Array(n - 1);
  for (let i = 0; i < n; i++) {
    let c = 0;
    for (let j = 0; j < n; j++) if (j !== i) row[c++] = S[i * n + j];
    row.sort();
    let s = 0;
    const kk = Math.min(k, n - 1);
    for (let t = 0; t < kk; t++) s += row[n - 2 - t];
    r[i] = s / kk;
  }
  return r;
}

// CSLS neighbor lists: rank j for i by 2·S(i,j) − r(i) − r(j). Hubs (high r)
// are pushed down, so no single image crowds every neighborhood.
export function cslsNeighbors(S: Float64Array, n: number, r: Float64Array, k: number): number[][] {
  const out: number[][] = [];
  const idx = Array.from({ length: n }, (_, j) => j);
  for (let i = 0; i < n; i++) {
    const score = (j: number) => 2 * S[i * n + j] - r[i] - r[j];
    out.push(idx.filter((j) => j !== i).sort((a, b) => score(b) - score(a)).slice(0, k));
  }
  return out;
}

// ---- Ward clustering (nearest-neighbor chain, O(n²)) ---------------------------

interface Merge {
  a: number;
  b: number;
  height: number;
}

function wardMerges(X: Vec[]): Merge[] {
  const n = X.length;
  const D = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      let s = 0;
      for (let d = 0; d < X[i].length; d++) {
        const t = X[i][d] - X[j][d];
        s += t * t;
      }
      D[i * n + j] = D[j * n + i] = s;
    }
  }
  const size = new Float64Array(n).fill(1);
  const active = new Uint8Array(n).fill(1);
  let remaining = n;
  const chain: number[] = [];
  const merges: Merge[] = [];
  while (remaining > 1) {
    if (chain.length === 0) chain.push(active.indexOf(1));
    const a = chain[chain.length - 1];
    const prev = chain.length > 1 ? chain[chain.length - 2] : -1;
    let b = -1;
    let best = Infinity;
    for (let x = 0; x < n; x++) {
      if (!active[x] || x === a) continue;
      const d = D[a * n + x];
      if (d < best || (d === best && x === prev)) {
        best = d;
        b = x;
      }
    }
    if (b === prev) {
      chain.pop();
      chain.pop();
      merges.push({ a, b, height: best });
      // Lance-Williams update for Ward; the merged cluster lives in slot a.
      for (let l = 0; l < n; l++) {
        if (!active[l] || l === a || l === b) continue;
        const nl = size[l];
        const d =
          ((size[a] + nl) * D[a * n + l] + (size[b] + nl) * D[b * n + l] - nl * best) /
          (size[a] + size[b] + nl);
        D[a * n + l] = D[l * n + a] = d;
      }
      size[a] += size[b];
      active[b] = 0;
      remaining--;
    } else {
      chain.push(b);
    }
  }
  // Ward is reducible, so sorting by height (stable) keeps every merge after
  // the merges it depends on.
  return merges.map((m, i) => ({ ...m, i })).sort((x, y) => x.height - y.height || x.i - y.i);
}

function cut(merges: Merge[], n: number, k: number): Int32Array {
  const parent = Int32Array.from({ length: n }, (_, i) => i);
  const find = (x: number): number => {
    while (parent[x] !== x) x = parent[x] = parent[parent[x]];
    return x;
  };
  for (let m = 0; m < Math.max(0, n - k); m++) parent[find(merges[m].b)] = find(merges[m].a);
  const ids = new Map<number, number>();
  const out = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    const root = find(i);
    if (!ids.has(root)) ids.set(root, ids.size);
    out[i] = ids.get(root)!;
  }
  return out;
}

export interface Hierarchy {
  coarse: Int32Array; // per item
  fine: Int32Array; // per item
  parentOf: Int32Array; // per fine cluster → coarse cluster
  core: Uint8Array; // 1 = well-connected, 0 = attached to its nearest core image
}

// Two nested cuts of one Ward tree. Loose images (lowest local density) sit
// out of the tree and join the cluster of their nearest well-connected image,
// so they cannot form a grab-bag cluster of leftovers.
export function buildHierarchy(
  F: Vec[],
  S: Float64Array,
  density: Float64Array,
  kCoarse: number,
  kFine: number,
  coreFraction: number
): Hierarchy {
  const n = F.length;
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => density[b] - density[a]);
  const coreIdx = order.slice(0, Math.max(kFine, Math.round(n * coreFraction))).sort((a, b) => a - b);
  const core = new Uint8Array(n);
  for (const i of coreIdx) core[i] = 1;
  const merges = wardMerges(coreIdx.map((i) => F[i]));
  const cc = cut(merges, coreIdx.length, kCoarse);
  const cf = cut(merges, coreIdx.length, kFine);
  const coarse = new Int32Array(n);
  const fine = new Int32Array(n);
  coreIdx.forEach((i, t) => {
    coarse[i] = cc[t];
    fine[i] = cf[t];
  });
  for (let i = 0; i < n; i++) {
    if (core[i]) continue;
    let best = -1;
    let bestS = -Infinity;
    for (const j of coreIdx) {
      if (S[i * n + j] > bestS) {
        bestS = S[i * n + j];
        best = j;
      }
    }
    coarse[i] = coarse[best];
    fine[i] = fine[best];
  }
  const parentOf = new Int32Array(kFine);
  for (let i = 0; i < n; i++) parentOf[fine[i]] = coarse[i];
  return { coarse, fine, parentOf, core };
}

// ---- 3D layout ------------------------------------------------------------------

// Top principal components through the n×n Gram matrix (power iteration with
// deflation). Used as the UMAP starting point so the global arrangement is
// stable instead of depending on a random start.
function pca(F: Vec[], dims: number): number[][] {
  const n = F.length;
  const m = meanVector(F);
  const C = F.map((v) => v.map((x, i) => x - m[i]));
  const G = new Float64Array(n * n);
  for (let i = 0; i < n; i++) for (let j = i; j < n; j++) G[i * n + j] = G[j * n + i] = dot(C[i], C[j]);
  const rand = mulberry32(7);
  const out: number[][] = Array.from({ length: n }, () => []);
  for (let d = 0; d < dims; d++) {
    let v = Float64Array.from({ length: n }, () => rand() - 0.5);
    let lambda = 0;
    for (let it = 0; it < 300; it++) {
      const w = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        let s = 0;
        for (let j = 0; j < n; j++) s += G[i * n + j] * v[j];
        w[i] = s;
      }
      lambda = Math.hypot(...w);
      v = w.map((x) => x / (lambda || 1));
    }
    for (let i = 0; i < n; i++) out[i].push(v[i] * Math.sqrt(lambda));
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) G[i * n + j] -= lambda * v[i] * v[j];
  }
  return out;
}

export interface LayoutOptions {
  neighbors: number;
  minDist: number;
  spread: number;
  epochs: number;
  clusterWeight: number; // 0 = pure similarity; higher shortens and adds same-cluster links
  seed: number;
}

// UMAP to 3D over the exact CSLS neighbor graph, nudged toward the clusters so
// each label sits over its own images. umap-js's built-in supervision cuts
// links between clusters almost completely (islands with empty space between
// them); here same-cluster links are only shortened and topped up, which keeps
// the cloud continuous.
export function layout3D(F: Vec[], S: Float64Array, nbrs: number[][], fine: Int32Array, o: LayoutOptions): number[][] {
  const n = F.length;
  const k = Math.min(o.neighbors, n - 1);
  const minSame = Math.round(k * o.clusterWeight);
  const knnIndices = nbrs.map((row, i) => {
    const list = row.slice(0, k - 1);
    const same = list.filter((j) => fine[j] === fine[i]).length;
    if (same < minSame) {
      // Replace the weakest cross-cluster links with the closest missing members.
      const extra = row.slice(k - 1).filter((j) => fine[j] === fine[i]).slice(0, minSame - same);
      for (const j of extra) {
        const drop = list.map((x, t) => [x, t]).reverse().find(([x]) => fine[x] !== fine[i]);
        if (drop) list[drop[1]] = j;
      }
    }
    return [i, ...list];
  });
  const knnDistances = knnIndices.map((row, i) =>
    row.map((j) => {
      if (j === i) return 0;
      const d = Math.max(0, 1 - S[i * n + j]);
      return fine[j] === fine[i] ? d * (1 - o.clusterWeight) : d;
    })
  );
  // Keep each row sorted by distance, as UMAP expects.
  knnIndices.forEach((row, i) => {
    const order = row.map((_, t) => t).sort((a, b) => knnDistances[i][a] - knnDistances[i][b]);
    knnIndices[i] = order.map((t) => row[t]);
    knnDistances[i] = order.map((t) => knnDistances[i][t]);
  });
  const umap = new UMAP({
    nComponents: 3,
    nNeighbors: k,
    minDist: o.minDist,
    spread: o.spread,
    nEpochs: o.epochs,
    random: mulberry32(o.seed),
  });
  umap.setPrecomputedKNN(knnIndices, knnDistances);
  const epochs = umap.initializeFit(F.map((v) => Array.from(v)));
  // Replace umap-js's random start with PCA, scaled the way umap-learn does.
  const init = pca(F, 3);
  let maxAbs = 1e-9;
  for (const p of init) for (const x of p) maxAbs = Math.max(maxAbs, Math.abs(x));
  const embedding = (umap as unknown as { embedding: number[][] }).embedding;
  for (let i = 0; i < n; i++) for (let d = 0; d < 3; d++) embedding[i][d] = (init[i][d] / maxAbs) * 10;
  for (let e = 0; e < epochs; e++) umap.step();
  const Y = umap.getEmbedding();
  // Center, then scale so 90% of the images sit inside the unit sphere.
  const c = [0, 1, 2].map((d) => Y.reduce((s, p) => s + p[d], 0) / n);
  const radii = Y.map((p) => Math.hypot(p[0] - c[0], p[1] - c[1], p[2] - c[2])).sort((a, b) => a - b);
  const scale = 1 / (radii[Math.floor(0.9 * (n - 1))] || 1);
  return Y.map((p) => p.map((x, d) => (x - c[d]) * scale));
}

// ---- zero-shot naming -------------------------------------------------------------

// A[i][t] = cosine of image i with term t (SigLIP image vs text embedding).
export type Affinity = Float64Array[];

export function affinity(images: Vec[], terms: Vec[]): Affinity {
  return images.map((v) => Float64Array.from(terms, (t) => dot(v, t)));
}

function columnStats(A: Affinity, t: number): { mean: number; std: number } {
  let s = 0;
  let s2 = 0;
  for (const row of A) {
    s += row[t];
    s2 += row[t] * row[t];
  }
  const mean = s / A.length;
  return { mean, std: Math.sqrt(Math.max(1e-12, s2 / A.length - mean * mean)) };
}

// Per-image tag score: half "distinctive for this image compared with the
// whole board" (column z-score), half "fits this image best among all terms"
// (row z-score).
export function tagScores(A: Affinity): Float64Array[] {
  const T = A[0].length;
  const cols = Array.from({ length: T }, (_, t) => columnStats(A, t));
  return A.map((row) => {
    let s = 0;
    let s2 = 0;
    for (const x of row) {
      s += x;
      s2 += x * x;
    }
    const mean = s / T;
    const std = Math.sqrt(Math.max(1e-12, s2 / T - mean * mean));
    return Float64Array.from(row, (x, t) => 0.5 * ((x - cols[t].mean) / cols[t].std) + 0.5 * ((x - mean) / std));
  });
}

export function topTerms(scores: Float64Array, k: number): number[] {
  return Array.from(scores.keys()).sort((a, b) => scores[b] - scores[a]).slice(0, k);
}

// Each image votes for its own best terms; c-TF-IDF then favors terms that
// many members vote for but few images on the rest of the board do.
function voteMatrix(tags: Float64Array[], weights: number[]): Float64Array[] {
  return tags.map((s) => {
    const v = new Float64Array(s.length);
    topTerms(s, weights.length).forEach((t, r) => (v[t] = weights[r]));
    return v;
  });
}

function ctfidf(V: Float64Array[], members: number[]): Float64Array {
  const T = V[0].length;
  const out = new Float64Array(T);
  for (let t = 0; t < T; t++) {
    let df = 0;
    for (const row of V) if (row[t] > 0) df++;
    let s = 0;
    for (const i of members) s += V[i][t];
    out[t] = (s / Math.max(1, members.length)) * Math.log(1 + V.length / (1 + df));
  }
  return out;
}

// Per term: AUC of members against the reference set (rank-based, so one
// outlier image cannot carry a term), times sqrt(share of members in the
// board's top 20% for that term).
function aucCoverage(A: Affinity, members: number[], reference: number[], top20: Float64Array): Float64Array {
  const T = A[0].length;
  const out = new Float64Array(T);
  for (let t = 0; t < T; t++) {
    const ref = reference.map((i) => A[i][t]).sort((a, b) => a - b);
    let wins = 0;
    let covered = 0;
    for (const i of members) {
      const x = A[i][t];
      let lo = 0;
      let hi = ref.length;
      while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (ref[mid] < x) lo = mid + 1;
        else hi = mid;
      }
      let ties = 0;
      while (lo + ties < ref.length && ref[lo + ties] === x) ties++;
      wins += lo + ties / 2;
      if (x >= top20[t]) covered++;
    }
    const auc = members.length && ref.length ? wins / (members.length * ref.length) : 0.5;
    out[t] = auc * Math.sqrt(covered / Math.max(1, members.length));
  }
  return out;
}

function percentile80(A: Affinity): Float64Array {
  const T = A[0].length;
  return Float64Array.from({ length: T }, (_, t) => {
    const col = A.map((row) => row[t]).sort((a, b) => a - b);
    return col[Math.floor(0.8 * (col.length - 1))];
  });
}

function normalizeMax(x: Float64Array): Float64Array {
  let m = 1e-9;
  for (const v of x) m = Math.max(m, v);
  return x.map((v) => v / m);
}

// Greedy one-to-one assignment: best (cluster, term) pair first, so two
// clusters never share a name.
function assignUnique(scores: Float64Array[]): number[] {
  const picked = new Array<number>(scores.length).fill(-1);
  const usedTerms = new Set<number>();
  for (let round = 0; round < scores.length; round++) {
    let best = -Infinity;
    let bc = -1;
    let bt = -1;
    scores.forEach((row, c) => {
      if (picked[c] >= 0) return;
      row.forEach((v, t) => {
        if (!usedTerms.has(t) && v > best) {
          best = v;
          bc = c;
          bt = t;
        }
      });
    });
    picked[bc] = bt;
    usedTerms.add(bt);
  }
  return picked;
}

export interface Naming {
  coarse: number[]; // term index into the broad vocabulary, per coarse cluster
  fine: number[]; // term index into the specific vocabulary, per fine cluster
}

export function nameClusters(
  h: Hierarchy,
  specific: Affinity,
  broad: Affinity,
  specificTerms: Vec[],
  broadTerms: Vec[]
): Naming {
  const n = h.fine.length;
  const kc = Math.max(...h.coarse) + 1;
  const kf = Math.max(...h.fine) + 1;
  const all = Array.from({ length: n }, (_, i) => i);
  const coreMembers = (pred: (i: number) => boolean) => all.filter((i) => h.core[i] && pred(i));

  const voteSpecific = voteMatrix(tagScores(specific), [1, 0.7, 0.5, 0.35, 0.25]);
  const voteBroad = voteMatrix(tagScores(broad), [1, 0.6, 0.35]);
  const topSpecific = percentile80(specific);
  const topBroad = percentile80(broad);

  const fineScores: Float64Array[] = [];
  for (let f = 0; f < kf; f++) {
    const members = coreMembers((i) => h.fine[i] === f);
    const everyoneElse = all.filter((i) => h.fine[i] !== f);
    const siblings = all.filter((i) => h.fine[i] !== f && h.coarse[i] === h.parentOf[f]);
    const global = aucCoverage(specific, members, everyoneElse, topSpecific);
    // Contrast with siblings too, so sub-groups of one region get names that
    // tell them apart rather than the region's shared trait.
    const local = siblings.length >= 3 ? aucCoverage(specific, members, siblings, topSpecific) : global;
    const auc = global.map((g, t) => 0.65 * g + 0.35 * local[t]);
    const votes = ctfidf(voteSpecific, all.filter((i) => h.fine[i] === f));
    const a = normalizeMax(auc);
    const v = normalizeMax(votes);
    fineScores.push(a.map((x, t) => 0.5 * x + 0.5 * v[t]));
  }
  const fine = assignUnique(fineScores);

  // Top-level groups: half image evidence, half the text-space neighborhood of
  // their children's names (weighted by size).
  const meanSpecific = meanVector(specificTerms);
  const coarseScores: Float64Array[] = [];
  for (let c = 0; c < kc; c++) {
    const members = coreMembers((i) => h.coarse[i] === c);
    const others = all.filter((i) => h.coarse[i] !== c);
    const image = normalizeMax(aucCoverage(broad, members, others, topBroad));
    const votes = normalizeMax(ctfidf(voteBroad, all.filter((i) => h.coarse[i] === c)));
    const childMix = new Float32Array(specificTerms[0].length);
    for (let f = 0; f < kf; f++) {
      if (h.parentOf[f] !== c) continue;
      const size = all.filter((i) => h.fine[i] === f).length;
      const tv = specificTerms[fine[f]];
      for (let d = 0; d < childMix.length; d++) childMix[d] += size * tv[d];
    }
    const mixNorm = Math.sqrt(dot(childMix, childMix)) || 1;
    const text = normalizeMax(
      Float64Array.from(broadTerms, (b) => Math.max(0, dot(b, childMix) / mixNorm - dot(b, meanSpecific)))
    );
    coarseScores.push(image.map((x, t) => 0.25 * x + 0.25 * votes[t] + 0.5 * text[t]));
  }
  const coarse = assignUnique(coarseScores);
  return { coarse, fine };
}
