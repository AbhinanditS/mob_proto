// Converge mode — pure logic (no React).
//
// Goal: from a diverse pool of candidate images, force pairwise "which do you
// want your portfolio to look like?" choices, score them with a
// Bradley-Terry / Elo model, then reveal the structure of what was chosen.
//
// All vectors here are the existing SigLIP embeddings from embeddings.json,
// already L2-normalized — so a dot product IS cosine similarity. Nothing is
// recomputed; we only read these vectors.

export interface ConvergeItem {
  id: string;
  vec: Float32Array;
}

export interface PoolEntry {
  id: string;
  vec: Float32Array;
  elo: number;          // live online score, updated every comparison
  appearances: number;
  wins: number;
  losses: number;
  strength: number;     // Bradley-Terry strength, filled in after the run
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
  normScore: number;    // 0..100 for display
  appearances: number;
  wins: number;
  losses: number;
  winRate: number;      // 0..1 (0 when never seen)
}

export interface ClusterLocation {
  clusterId: number;
  representativeId: string;   // member nearest the cluster centroid
  label: string;              // keywords mined from member filenames
  size: number;
  similarity: number;         // cosine(picks-centroid, cluster centroid)
}

export interface TensionPair {
  winnerId: string;           // the image you chose
  loserId: string;            // the image you rejected
  turn: number;
  winnerStrength: number;
  loserStrength: number;
  upset: number;              // loserStrength - winnerStrength (how big the contradiction)
}

export interface AnalysisResult {
  ranked: RankedRow[];
  centroid: Float32Array;     // strength-weighted centroid of the pool
  clusterLocation: ClusterLocation | null;
  inBetween: string[];        // 20-25 ids from the FULL dataset, never compared
  tensionPair: TensionPair | null;
}

// Tunables
export const POOL_SIZE = 28;
export const MAX_COMPARISONS = 20;
const ELO_START = 1500;
const ELO_K = 24;
const APPEAR_PENALTY = 40;     // Elo-points per prior appearance (coverage pressure)
const K_CLUSTERS = 8;
const KMEANS_SEED = 1234567;   // fixed → deterministic, frozen clustering
const BT_PRIOR = 1.0;          // virtual win/loss vs a mean-strength phantom
const IN_BETWEEN_COUNT = 24;

// ---------------------------------------------------------------------------
// Vector math
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

// Deterministic PRNG (mulberry32) so k-means is identical across reloads.
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

// Greedily pick `n` items that are maximally spread across embedding space, so
// the forced choices span the full aesthetic range. Fresh seed each run.
export function farthestPointSample(items: ConvergeItem[], n: number): ConvergeItem[] {
  const count = Math.min(n, items.length);
  if (count <= 0) return [];

  const selected: ConvergeItem[] = [];
  const seedIdx = Math.floor(Math.random() * items.length);
  selected.push(items[seedIdx]);

  // minDist[i] = min cosine distance (1 - dot) from item i to the chosen set.
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
    // Update min distances against the newly added point.
    for (let i = 0; i < items.length; i++) {
      if (minDist[i] === -Infinity) continue;
      const d = 1 - dot(items[i].vec, added.vec);
      if (d < minDist[i]) minDist[i] = d;
    }
  }

  return selected;
}

// ---------------------------------------------------------------------------
// Comparison engine — Elo + adaptive pairing
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
    elo: ELO_START,
    appearances: 0,
    wins: 0,
    losses: 0,
    strength: 1,
  }));
  const byId = new Map(pool.map((p) => [p.id, p]));
  const shownPairs = new Set<string>();
  const comparisons: Comparison[] = [];

  // Pick the next pair: among pairs never shown, minimize
  //   |elo_i - elo_j| + APPEAR_PENALTY * (appearances_i + appearances_j)
  // i.e. closest-rated (most uncertain → most informative) AND fewest seen.
  function pickNextPair(prevKey: string | null): [PoolEntry, PoolEntry] | null {
    let best: [PoolEntry, PoolEntry] | null = null;
    let bestCost = Infinity;
    let fallback: [PoolEntry, PoolEntry] | null = null;
    let fallbackCost = Infinity;

    for (let i = 0; i < pool.length; i++) {
      for (let j = i + 1; j < pool.length; j++) {
        const a = pool[i];
        const b = pool[j];
        const key = pairKey(a.id, b.id);
        if (shownPairs.has(key)) continue;
        const cost =
          Math.abs(a.elo - b.elo) +
          APPEAR_PENALTY * (a.appearances + b.appearances);
        // Track an overall best (used if every remaining pair was the prev one).
        if (cost < fallbackCost) {
          fallbackCost = cost;
          fallback = [a, b];
        }
        // Prefer not to immediately repeat the previous visual pairing.
        if (key === prevKey) continue;
        if (cost < bestCost) {
          bestCost = cost;
          best = [a, b];
        }
      }
    }
    return best ?? fallback;
  }

  function eloUpdate(winner: PoolEntry, loser: PoolEntry): void {
    const expWin = 1 / (1 + Math.pow(10, (loser.elo - winner.elo) / 400));
    winner.elo += ELO_K * (1 - expWin);
    loser.elo -= ELO_K * (1 - expWin);
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
      eloUpdate(winner, loser);

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

// Joint fit over the full comparison log. A small prior (virtual win + loss vs
// a phantom of mean strength) regularizes undefeated/winless items and keeps
// strengths finite even when the comparison graph is sparse or disconnected —
// which it will be with ~28 images and 20 rounds.
export function fitBradleyTerry(
  pool: PoolEntry[],
  comparisons: Comparison[],
  iterations = 100
): void {
  const n = pool.length;
  if (n === 0) return;

  const idx = new Map(pool.map((p, i) => [p.id, i]));
  const strength = new Float64Array(n).fill(1);

  // Total wins per item, and pairwise comparison counts n_ij (symmetric).
  const wins = new Float64Array(n);
  const pairCounts = new Map<number, number>(); // key i*n+j (i<j) -> count
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

  for (let iter = 0; iter < iterations; iter++) {
    const next = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      // Denominator: real comparisons + prior games vs a phantom of strength 1.
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
    // Normalize so the geometric mean of strengths is 1 (identifiability).
    let logSum = 0;
    for (let i = 0; i < n; i++) logSum += Math.log(next[i]);
    const geo = Math.exp(logSum / n);
    for (let i = 0; i < n; i++) strength[i] = next[i] / geo;
  }

  for (let i = 0; i < n; i++) pool[i].strength = strength[i];
}

// ---------------------------------------------------------------------------
// Spherical k-means over the FULL dataset (deterministic, cached once)
// ---------------------------------------------------------------------------

interface ClusterModel {
  centroids: Float32Array[];
  assignments: number[];   // cluster index per item, parallel to the items array
  itemIds: string[];
}

let clusterCache: ClusterModel | null = null;
let clusterCacheKey = "";

// Cosine k-means: vectors are unit-norm, so assignment = argmax dot, and each
// centroid is the L2-normalized mean of its members. Seeded → frozen geometry.
export function getClusters(items: ConvergeItem[], k = K_CLUSTERS): ClusterModel {
  const cacheKey = `${items.length}:${k}`;
  if (clusterCache && clusterCacheKey === cacheKey) return clusterCache;

  const n = items.length;
  const dim = n > 0 ? items[0].vec.length : 0;
  const rng = mulberry32(KMEANS_SEED);

  // k-means++ style seeding with the seeded PRNG.
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
    // Assign.
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
    // Recompute centroids as normalized means.
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
      if (counts[c] === 0) continue; // keep prior centroid for empty clusters
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

// Filenames look like: `001_733734964335..._This_may_contain_<caption>`.
// Strip the leading numeric ids and mine frequent caption tokens.
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

  // Joint Bradley-Terry fit for the accurate final ranking.
  fitBradleyTerry(pool, comparisons);

  // 1. Ranked list with normalized 0..100 scores.
  const sorted = [...pool].sort((a, b) => b.strength - a.strength);
  const logs = sorted.map((p) => Math.log(p.strength));
  const minLog = Math.min(...logs);
  const maxLog = Math.max(...logs);
  const span = maxLog - minLog || 1;
  const ranked: RankedRow[] = sorted.map((p, i) => ({
    rank: i + 1,
    id: p.id,
    strength: p.strength,
    normScore: ((Math.log(p.strength) - minLog) / span) * 100,
    appearances: p.appearances,
    wins: p.wins,
    losses: p.losses,
    winRate: p.appearances > 0 ? p.wins / p.appearances : 0,
  }));

  // 2. Strength-weighted centroid of the pool (top picks dominate).
  const dim = pool.length > 0 ? pool[0].vec.length : 0;
  const acc = new Float64Array(dim);
  let wsum = 0;
  for (const p of pool) {
    const w = p.strength;
    wsum += w;
    for (let d = 0; d < dim; d++) acc[d] += w * p.vec[d];
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
    // Members of the winning cluster + its representative image.
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
