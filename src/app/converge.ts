// Converge mode — pure logic (no React).
//
// From a diverse pool of candidate images, force pairwise "which do you want
// your portfolio to look like?" choices, score them, then reveal the structure
// of what was chosen.
//
// Scoring is deliberately deep:
//   - An online TrueSkill-style model (mean mu + uncertainty sigma per image),
//     updated after every comparison. The uncertainty drives both pairing and
//     the displayed confidence.
//   - Adaptive pairing that maximizes information AND visual difficulty: it
//     prefers near-coin-flip matchups between visually similar images, so each
//     choice is genuinely hard and maximally informative.
//   - A final joint Bradley-Terry MAP fit over the whole comparison log for the
//     most accurate point ranking, with embedding-based propagation so the
//     under-compared tail is ranked by what its neighbors achieved, plus
//     Fisher-information confidence intervals.
//
// All vectors are the existing SigLIP embeddings (already L2-normalized, so a
// dot product IS cosine similarity). Nothing is recomputed.

export interface ConvergeItem {
  id: string;
  vec: Float32Array;
}

export interface PoolEntry {
  id: string;
  vec: Float32Array;
  mu: number;           // TrueSkill skill mean (online)
  sigma: number;        // TrueSkill skill uncertainty (online)
  appearances: number;
  wins: number;
  losses: number;
  strength: number;     // final Bradley-Terry strength (filled after the run)
}

export interface Comparison {
  leftId: string;
  rightId: string;
  winnerId: string;
  loserId: string;
  turn: number;
}

export interface RankedRow {
  rank: number;
  id: string;
  strength: number;
  normScore: number;    // 0..100 point estimate
  ciLow: number;        // 0..100 lower confidence bound
  ciHigh: number;       // 0..100 upper confidence bound
  appearances: number;
  wins: number;
  losses: number;
  winRate: number;      // 0..1 (0 when never seen)
}

export interface ClusterLocation {
  clusterId: number;
  representativeId: string;
  label: string;
  size: number;
  similarity: number;
}

export interface TensionPair {
  winnerId: string;
  loserId: string;
  turn: number;
  winnerStrength: number;
  loserStrength: number;
  upset: number;
}

export interface AnalysisResult {
  ranked: RankedRow[];
  centroid: Float32Array;
  clusterLocation: ClusterLocation | null;
  inBetween: string[];
  tensionPair: TensionPair | null;
}

// Tunables
export const POOL_SIZE = 26;
export const MAX_COMPARISONS = 26;

// TrueSkill parameters (standard defaults, on a 0..50 scale).
const MU0 = 25;
const SIGMA0 = 25 / 3;
const BETA2 = (SIGMA0 / 2) * (SIGMA0 / 2); // performance variance
const TAU2 = (SIGMA0 / 100) * (SIGMA0 / 100); // per-game dynamics
const PAIR_EMB_WEIGHT = 0.5; // how strongly to prefer visually similar pairs

// Bradley-Terry / propagation / clustering
const BT_PRIOR = 1.0;        // virtual win/loss vs a mean-strength phantom
const BT_ITERS = 200;
const BT_RIDGE = 0.25;       // Normal(0, 1/ridge) prior on log-strength → bounds SEs
const PROP_K0 = 2.5;         // appearances pseudo-count for the propagation blend
const PROP_TEMP = 0.08;      // softmax temperature over cosine in propagation
const K_CLUSTERS = 8;
const KMEANS_SEED = 1234567;
const IN_BETWEEN_COUNT = 24;

// ---------------------------------------------------------------------------
// Vector + scalar math
// ---------------------------------------------------------------------------

export function dot(a: Float32Array, b: Float32Array): number {
  let s = 0;
  const len = a.length;
  for (let i = 0; i < len; i++) s += a[i] * b[i];
  return s;
}

function l2normalize(v: Float32Array): Float32Array {
  let norm = 0;
  for (let i = 0; i < v.length; i++) norm += v[i] * v[i];
  norm = Math.sqrt(norm) || 1;
  const out = new Float32Array(v.length);
  for (let i = 0; i < v.length; i++) out[i] = v[i] / norm;
  return out;
}

// Abramowitz-Stegun erf → standard normal pdf/cdf.
function erf(x: number): number {
  const sign = x < 0 ? -1 : 1;
  const ax = Math.abs(x);
  const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741,
    a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
  const t = 1 / (1 + p * ax);
  const y = 1 - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * Math.exp(-ax * ax);
  return sign * y;
}
const INV_SQRT_2PI = 1 / Math.sqrt(2 * Math.PI);
function normCdf(x: number): number {
  return 0.5 * (1 + erf(x / Math.SQRT2));
}
function normPdf(x: number): number {
  return INV_SQRT_2PI * Math.exp(-0.5 * x * x);
}
// TrueSkill truncated-Gaussian helpers for a win (no draws).
function vWin(t: number): number {
  const c = normCdf(t);
  return c < 1e-9 ? -t : normPdf(t) / c; // asymptote for extreme upsets
}
function wWin(t: number): number {
  const v = vWin(t);
  return v * (v + t);
}
function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

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

// ---------------------------------------------------------------------------
// Candidate pool — farthest-point sampling for a diverse spread
// ---------------------------------------------------------------------------

export function farthestPointSample(items: ConvergeItem[], n: number): ConvergeItem[] {
  const count = Math.min(n, items.length);
  if (count <= 0) return [];

  const selected: ConvergeItem[] = [];
  const seedIdx = Math.floor(Math.random() * items.length);
  selected.push(items[seedIdx]);

  const minDist = new Float32Array(items.length);
  for (let i = 0; i < items.length; i++) {
    minDist[i] = 1 - dot(items[i].vec, items[seedIdx].vec);
  }
  minDist[seedIdx] = -Infinity;

  while (selected.length < count) {
    let best = -1;
    let bestDist = -Infinity;
    for (let i = 0; i < items.length; i++) {
      if (minDist[i] > bestDist) {
        bestDist = minDist[i];
        best = i;
      }
    }
    if (best < 0) break;
    selected.push(items[best]);
    const added = items[best];
    minDist[best] = -Infinity;
    for (let i = 0; i < items.length; i++) {
      if (minDist[i] === -Infinity) continue;
      const d = 1 - dot(items[i].vec, added.vec);
      if (d < minDist[i]) minDist[i] = d;
    }
  }

  return selected;
}

// ---------------------------------------------------------------------------
// Comparison engine — online TrueSkill + information-maximizing pairing
// ---------------------------------------------------------------------------

function pairKey(a: string, b: string): string {
  return a < b ? `${a}|${b}` : `${b}|${a}`;
}

export interface ConvergeEngine {
  pool: PoolEntry[];
  byId: Map<string, PoolEntry>;
  comparisons: Comparison[];
  maxComparisons: number;
  currentPair: [PoolEntry, PoolEntry] | null;
  isDone(): boolean;
  choose(winnerId: string): void;
}

export function createEngine(
  poolItems: ConvergeItem[],
  maxComparisons: number = MAX_COMPARISONS
): ConvergeEngine {
  const pool: PoolEntry[] = poolItems.map((it) => ({
    id: it.id,
    vec: it.vec,
    mu: MU0,
    sigma: SIGMA0,
    appearances: 0,
    wins: 0,
    losses: 0,
    strength: 1,
  }));
  const byId = new Map(pool.map((p) => [p.id, p]));
  const shownPairs = new Set<string>();
  const comparisons: Comparison[] = [];

  // Predicted P(a beats b) under the current skill beliefs.
  function predictP(a: PoolEntry, b: PoolEntry): number {
    const c = Math.sqrt(2 * BETA2 + a.sigma * a.sigma + b.sigma * b.sigma);
    return normCdf((a.mu - b.mu) / c);
  }

  // Pair value = expected information × visual difficulty.
  //   p(1-p)            → highest for a near coin-flip (a genuinely hard choice)
  //   sigma_a²+sigma_b² → highest for under-explored images (learn the most)
  //   (1 + w·cos)       → bias toward visually similar pairs (subtle choices)
  function pairValue(a: PoolEntry, b: PoolEntry): number {
    const p = predictP(a, b);
    const info = p * (1 - p) * (a.sigma * a.sigma + b.sigma * b.sigma);
    const cos = dot(a.vec, b.vec);
    return info * (1 + PAIR_EMB_WEIGHT * Math.max(0, cos));
  }

  function pickNextPair(prevKey: string | null): [PoolEntry, PoolEntry] | null {
    let best: [PoolEntry, PoolEntry] | null = null;
    let bestVal = -Infinity;
    let fallback: [PoolEntry, PoolEntry] | null = null;
    let fallbackVal = -Infinity;

    for (let i = 0; i < pool.length; i++) {
      for (let j = i + 1; j < pool.length; j++) {
        const a = pool[i];
        const b = pool[j];
        const key = pairKey(a.id, b.id);
        if (shownPairs.has(key)) continue;
        const val = pairValue(a, b);
        if (val > fallbackVal) {
          fallbackVal = val;
          fallback = [a, b];
        }
        if (key === prevKey) continue; // avoid an immediate visual repeat
        if (val > bestVal) {
          bestVal = val;
          best = [a, b];
        }
      }
    }
    return best ?? fallback;
  }

  function trueskillUpdate(winner: PoolEntry, loser: PoolEntry): void {
    const sw2 = winner.sigma * winner.sigma + TAU2;
    const sl2 = loser.sigma * loser.sigma + TAU2;
    const c2 = 2 * BETA2 + sw2 + sl2;
    const c = Math.sqrt(c2);
    const t = (winner.mu - loser.mu) / c;
    const v = vWin(t);
    const w = wWin(t);
    winner.mu += (sw2 / c) * v;
    loser.mu -= (sl2 / c) * v;
    winner.sigma = Math.sqrt(Math.max(1e-6, sw2 * (1 - (sw2 / c2) * w)));
    loser.sigma = Math.sqrt(Math.max(1e-6, sl2 * (1 - (sl2 / c2) * w)));
  }

  const engine: ConvergeEngine = {
    pool,
    byId,
    comparisons,
    maxComparisons,
    currentPair: pickNextPair(null),
    isDone() {
      return this.comparisons.length >= this.maxComparisons || this.currentPair === null;
    },
    choose(winnerId: string) {
      if (!this.currentPair) return;
      const [a, b] = this.currentPair;
      const winner = winnerId === a.id ? a : winnerId === b.id ? b : null;
      if (!winner) return;
      const loser = winner === a ? b : a;

      winner.appearances++;
      loser.appearances++;
      winner.wins++;
      loser.losses++;
      trueskillUpdate(winner, loser);

      const turn = this.comparisons.length;
      this.comparisons.push({
        leftId: a.id,
        rightId: b.id,
        winnerId: winner.id,
        loserId: loser.id,
        turn,
      });
      const key = pairKey(a.id, b.id);
      shownPairs.add(key);

      this.currentPair = this.isDone() ? null : pickNextPair(key);
    },
  };

  return engine;
}

// ---------------------------------------------------------------------------
// Final ranking — Bayesian Bradley-Terry (minorization-maximization)
// ---------------------------------------------------------------------------

interface PairAgg {
  wins: Float64Array;                 // total wins per index
  pairCounts: Map<number, number>;    // key a*n+b (a<b) -> total comparisons
}

function aggregate(
  pool: PoolEntry[],
  comparisons: Comparison[]
): PairAgg {
  const n = pool.length;
  const idx = new Map(pool.map((p, i) => [p.id, i]));
  const wins = new Float64Array(n);
  const pairCounts = new Map<number, number>();
  for (const c of comparisons) {
    const wi = idx.get(c.winnerId);
    const li = idx.get(c.loserId);
    if (wi === undefined || li === undefined) continue;
    wins[wi] += 1;
    const i = Math.min(wi, li);
    const j = Math.max(wi, li);
    const key = i * n + j;
    pairCounts.set(key, (pairCounts.get(key) ?? 0) + 1);
  }
  return { wins, pairCounts };
}

// Joint fit over the full comparison log. A small prior (virtual win + loss vs
// a phantom of mean strength) regularizes undefeated/winless items and keeps
// strengths finite even when the comparison graph is sparse or disconnected.
// Returns geometric-mean-normalized strengths.
export function fitBradleyTerry(
  pool: PoolEntry[],
  comparisons: Comparison[],
  iterations = BT_ITERS
): Float64Array {
  const n = pool.length;
  const strength = new Float64Array(n).fill(1);
  if (n === 0) return strength;

  const { wins, pairCounts } = aggregate(pool, comparisons);

  for (let iter = 0; iter < iterations; iter++) {
    const next = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let denom = (2 * BT_PRIOR) / (strength[i] + 1);
      for (let j = 0; j < n; j++) {
        if (j === i) continue;
        const a = Math.min(i, j);
        const b = Math.max(i, j);
        const nij = pairCounts.get(a * n + b);
        if (!nij) continue;
        denom += nij / (strength[i] + strength[j]);
      }
      next[i] = (wins[i] + BT_PRIOR) / denom;
    }
    let logSum = 0;
    for (let i = 0; i < n; i++) logSum += Math.log(next[i]);
    const geo = Math.exp(logSum / n);
    for (let i = 0; i < n; i++) strength[i] = next[i] / geo;
  }

  return strength;
}

// Invert a small dense matrix via Gauss-Jordan (n is tiny here).
function invertMatrix(A: number[][]): number[][] {
  const n = A.length;
  const M = A.map((row, i) => {
    const aug = new Array(2 * n).fill(0);
    for (let j = 0; j < n; j++) aug[j] = row[j];
    aug[n + i] = 1;
    return aug;
  });
  for (let col = 0; col < n; col++) {
    let piv = col;
    let max = Math.abs(M[col][col]);
    for (let r = col + 1; r < n; r++) {
      if (Math.abs(M[r][col]) > max) {
        max = Math.abs(M[r][col]);
        piv = r;
      }
    }
    if (max < 1e-12) M[col][col] += 1e-9; // numerical guard
    if (piv !== col) {
      const tmp = M[piv];
      M[piv] = M[col];
      M[col] = tmp;
    }
    const d = M[col][col];
    for (let j = 0; j < 2 * n; j++) M[col][j] /= d;
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const f = M[r][col];
      if (f !== 0) for (let j = 0; j < 2 * n; j++) M[r][j] -= f * M[col][j];
    }
  }
  return M.map((row) => row.slice(n));
}

// Standard errors on the log-strengths from the Bradley-Terry Fisher
// information (a p(1-p)-weighted graph Laplacian, made positive-definite by the
// prior). Under-compared images get larger errors → wider confidence bands.
export function standardErrors(
  pool: PoolEntry[],
  comparisons: Comparison[],
  strength: Float64Array
): Float64Array {
  const n = pool.length;
  const se = new Float64Array(n);
  if (n === 0) return se;
  if (n === 1) {
    se[0] = 1;
    return se;
  }

  const theta = Array.from(strength, (s) => Math.log(s));
  const { pairCounts } = aggregate(pool, comparisons);
  const H: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));

  for (const [key, nij] of pairCounts) {
    const i = Math.floor(key / n);
    const j = key % n;
    const p = sigmoid(theta[i] - theta[j]);
    const g = nij * p * (1 - p);
    H[i][i] += g;
    H[j][j] += g;
    H[i][j] -= g;
    H[j][i] -= g;
  }
  // Prior: 2*alpha phantom games vs a fixed phantom at theta=0, plus a ridge
  // (Normal prior on theta) that bounds the variance of all-win/all-loss items.
  for (let i = 0; i < n; i++) {
    const p0 = sigmoid(theta[i]);
    H[i][i] += 2 * BT_PRIOR * p0 * (1 - p0) + BT_RIDGE;
  }

  const Hinv = invertMatrix(H);
  for (let i = 0; i < n; i++) se[i] = Math.sqrt(Math.max(0, Hinv[i][i]));
  return se;
}

// Embedding-based propagation: pull each image's log-strength toward a
// similarity-weighted average of the well-observed images. Heavily-compared
// images keep their own estimate; the sparse tail inherits its neighbors'.
export function propagateStrengths(
  pool: PoolEntry[],
  strength: Float64Array
): Float64Array {
  const n = pool.length;
  if (n === 0) return strength;

  const theta = Array.from(strength, (s) => Math.log(s));
  const reliability = pool.map((p) => p.appearances / (p.appearances + PROP_K0));
  const out = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    let wsum = 0;
    let acc = 0;
    for (let j = 0; j < n; j++) {
      if (j === i) continue;
      const w = Math.exp(dot(pool[i].vec, pool[j].vec) / PROP_TEMP) * reliability[j];
      wsum += w;
      acc += w * theta[j];
    }
    const thetaKernel = wsum > 0 ? acc / wsum : theta[i];
    const blend = reliability[i];
    out[i] = blend * theta[i] + (1 - blend) * thetaKernel;
  }

  // Re-normalize to geometric mean 1.
  let mean = 0;
  for (let i = 0; i < n; i++) mean += out[i];
  mean /= n;
  const result = new Float64Array(n);
  for (let i = 0; i < n; i++) result[i] = Math.exp(out[i] - mean);
  return result;
}

// ---------------------------------------------------------------------------
// Spherical k-means over the FULL dataset (deterministic, cached once)
// ---------------------------------------------------------------------------

interface ClusterModel {
  centroids: Float32Array[];
  assignments: number[];
  itemIds: string[];
}

let clusterCache: ClusterModel | null = null;
let clusterCacheKey = "";

export function getClusters(items: ConvergeItem[], k = K_CLUSTERS): ClusterModel {
  const cacheKey = `${items.length}:${k}`;
  if (clusterCache && clusterCacheKey === cacheKey) return clusterCache;

  const n = items.length;
  const dim = n > 0 ? items[0].vec.length : 0;
  const rng = mulberry32(KMEANS_SEED);

  const centroids: Float32Array[] = [];
  const first = Math.floor(rng() * n);
  centroids.push(new Float32Array(items[first].vec));
  const dist = new Float64Array(n).fill(Infinity);
  for (let c = 1; c < k; c++) {
    let sum = 0;
    for (let i = 0; i < n; i++) {
      const d = 1 - dot(items[i].vec, centroids[c - 1]);
      if (d < dist[i]) dist[i] = d;
      sum += dist[i];
    }
    let target = rng() * sum;
    let chosen = 0;
    for (let i = 0; i < n; i++) {
      target -= dist[i];
      if (target <= 0) {
        chosen = i;
        break;
      }
    }
    centroids.push(new Float32Array(items[chosen].vec));
  }

  const assignments = new Array<number>(n).fill(0);
  for (let iter = 0; iter < 50; iter++) {
    let changed = false;
    for (let i = 0; i < n; i++) {
      let best = 0;
      let bestSim = -Infinity;
      for (let c = 0; c < k; c++) {
        const sim = dot(items[i].vec, centroids[c]);
        if (sim > bestSim) {
          bestSim = sim;
          best = c;
        }
      }
      if (assignments[i] !== best) {
        assignments[i] = best;
        changed = true;
      }
    }
    const sums = Array.from({ length: k }, () => new Float64Array(dim));
    const counts = new Int32Array(k);
    for (let i = 0; i < n; i++) {
      const c = assignments[i];
      counts[c]++;
      const v = items[i].vec;
      const s = sums[c];
      for (let d = 0; d < dim; d++) s[d] += v[d];
    }
    for (let c = 0; c < k; c++) {
      if (counts[c] === 0) continue;
      const mean = new Float32Array(dim);
      for (let d = 0; d < dim; d++) mean[d] = sums[c][d] / counts[c];
      centroids[c] = l2normalize(mean);
    }
    if (!changed && iter > 0) break;
  }

  clusterCache = { centroids, assignments, itemIds: items.map((it) => it.id) };
  clusterCacheKey = cacheKey;
  return clusterCache;
}

// ---------------------------------------------------------------------------
// Filename → keyword label
// ---------------------------------------------------------------------------

const STOPWORDS = new Set([
  "this", "may", "contain", "a", "an", "the", "of", "on", "in", "with", "and",
  "to", "is", "are", "that", "it", "its", "be", "or", "for", "at", "by", "from",
  "as", "image", "photo", "picture", "next", "near", "some", "other",
]);

function labelFromIds(ids: string[], byId: Map<string, ConvergeItem>): string {
  const counts = new Map<string, number>();
  for (const id of ids) {
    if (!byId.has(id)) continue;
    const caption = id.replace(/^\d+_\d+_/, "").replace(/^this_may_contain_/i, "");
    const tokens = caption.toLowerCase().split(/[_\s]+/);
    const seen = new Set<string>();
    for (const tok of tokens) {
      const t = tok.replace(/[^a-z]/g, "");
      if (t.length < 3 || STOPWORDS.has(t) || seen.has(t)) continue;
      seen.add(t);
      counts.set(t, (counts.get(t) ?? 0) + 1);
    }
  }
  const top = [...counts.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, 4)
    .map(([t]) => t);
  return top.length ? top.join(" · ") : "—";
}

// ---------------------------------------------------------------------------
// Analysis (after the run)
// ---------------------------------------------------------------------------

export function analyze(
  engine: ConvergeEngine,
  allItems: ConvergeItem[]
): AnalysisResult {
  const { pool, comparisons } = engine;
  const n = pool.length;

  // Accurate joint point estimate, its confidence, then embedding propagation.
  const bt = fitBradleyTerry(pool, comparisons);
  const se = standardErrors(pool, comparisons, bt);
  const finalStrength = propagateStrengths(pool, bt);
  for (let i = 0; i < n; i++) pool[i].strength = finalStrength[i];

  // 1. Ranked list with normalized scores + confidence bands.
  const order = Array.from({ length: n }, (_, i) => i).sort(
    (a, b) => finalStrength[b] - finalStrength[a]
  );
  const logs = Array.from(finalStrength, (s) => Math.log(s));
  // The display scale comes from the Bradley-Terry spread (not the propagated,
  // compressed one) so confidence widths stay in honest, comparable units.
  const btLogs = Array.from(bt, (s) => Math.log(s));
  const minLog = Math.min(...btLogs);
  const maxLog = Math.max(...btLogs);
  const slope = 100 / (maxLog - minLog || 1);
  const clamp = (x: number) => Math.max(0, Math.min(100, x));
  const ranked: RankedRow[] = order.map((i, rankIdx) => {
    const p = pool[i];
    const normScore = clamp((logs[i] - minLog) * slope);
    const half = se[i] * slope;
    return {
      rank: rankIdx + 1,
      id: p.id,
      strength: finalStrength[i],
      normScore,
      ciLow: clamp(normScore - half),
      ciHigh: clamp(normScore + half),
      appearances: p.appearances,
      wins: p.wins,
      losses: p.losses,
      winRate: p.appearances > 0 ? p.wins / p.appearances : 0,
    };
  });

  // 2. Strength-weighted centroid of the pool (top picks dominate).
  const dim = n > 0 ? pool[0].vec.length : 0;
  const acc = new Float64Array(dim);
  let wsum = 0;
  for (let i = 0; i < n; i++) {
    const w = finalStrength[i];
    wsum += w;
    const v = pool[i].vec;
    for (let d = 0; d < dim; d++) acc[d] += w * v[d];
  }
  const centroidRaw = new Float32Array(dim);
  for (let d = 0; d < dim; d++) centroidRaw[d] = wsum > 0 ? acc[d] / wsum : 0;
  const centroid = l2normalize(centroidRaw);

  const allById = new Map(allItems.map((it) => [it.id, it]));

  // 3. Cluster location: which existing cluster the centroid falls in.
  let clusterLocation: ClusterLocation | null = null;
  if (dim > 0 && allItems.length > 0) {
    const model = getClusters(allItems);
    let bestC = 0;
    let bestSim = -Infinity;
    for (let c = 0; c < model.centroids.length; c++) {
      const sim = dot(centroid, model.centroids[c]);
      if (sim > bestSim) {
        bestSim = sim;
        bestC = c;
      }
    }
    const memberIds: string[] = [];
    let repId = "";
    let repSim = -Infinity;
    for (let i = 0; i < model.assignments.length; i++) {
      if (model.assignments[i] !== bestC) continue;
      const id = model.itemIds[i];
      memberIds.push(id);
      const item = allById.get(id);
      if (item) {
        const sim = dot(item.vec, model.centroids[bestC]);
        if (sim > repSim) {
          repSim = sim;
          repId = id;
        }
      }
    }
    clusterLocation = {
      clusterId: bestC,
      representativeId: repId || memberIds[0] || "",
      label: labelFromIds(memberIds, allById),
      size: memberIds.length,
      similarity: bestSim,
    };
  }

  // 4. "In between" surface: nearest images to the centroid from the FULL
  //    dataset that were never in the comparison pool.
  const poolIds = new Set(pool.map((p) => p.id));
  const inBetween = allItems
    .filter((it) => !poolIds.has(it.id))
    .map((it) => ({ id: it.id, s: dot(centroid, it.vec) }))
    .sort((a, b) => b.s - a.s)
    .slice(0, IN_BETWEEN_COUNT)
    .map((x) => x.id);

  // 5. Tension pair: the comparison whose winner most contradicts the final
  //    ranking (largest strength(loser) - strength(winner)).
  let tensionPair: TensionPair | null = null;
  let maxUpset = -Infinity;
  for (const c of comparisons) {
    const w = engine.byId.get(c.winnerId);
    const l = engine.byId.get(c.loserId);
    if (!w || !l) continue;
    const upset = l.strength - w.strength;
    if (upset > maxUpset) {
      maxUpset = upset;
      tensionPair = {
        winnerId: c.winnerId,
        loserId: c.loserId,
        turn: c.turn,
        winnerStrength: w.strength,
        loserStrength: l.strength,
        upset,
      };
    }
  }

  return { ranked, centroid, clusterLocation, inBetween, tensionPair };
}
