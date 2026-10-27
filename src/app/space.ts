// The image space at runtime: decodes src/data/space.json (built offline by
// scripts/precompute.ts) and ranks images. No React, no DOM.
//
// Two kinds of similarity:
//   - visual: the fused vector (SigLIP2 + DINOv3, each centered on the board's
//     mean), ranked with a CSLS hub penalty so a few "average" images cannot
//     crowd every result list;
//   - text: a SigLIP2 text vector against the raw SigLIP2 image vectors.

import raw from "@/data/space.json";
import { centered, decodeHalf, dot, fuse, normalize } from "@/lib/vectors";

export type Vec3 = [number, number, number];

export interface SpaceItem {
  id: string;
  thumb: string;
  w: number;
  h: number;
  kind?: "image" | "video";
  file?: string; // uploaded video source
  siglip: Float32Array; // L2-normalized SigLIP2 image vector, for text search
  fused: Float32Array; // visual vector, for image-to-image similarity
  hub: number; // mean similarity to its nearest images (CSLS penalty)
  textHub: number; // mean similarity to its best-matching vocabulary words
  pos: Vec3; // position in the 3D cloud
  cluster: number; // index of its fine cluster in `clusters`
  tags: string[];
}

export interface SpaceCluster {
  label: string;
  level: 0 | 1; // 0 = top-level group, 1 = cluster
  parent: number | null;
  size: number;
  pos: Vec3;
}

const MEAN_SIGLIP = decodeHalf(raw.means.siglip);
const MEAN_DINO = decodeHalf(raw.means.dino);
const WEIGHTS = [raw.weights.siglip, raw.weights.dino];

export const HUB_K = 10;

export function fusedVector(siglip: Float32Array, dino: Float32Array): Float32Array {
  return fuse([centered(siglip, MEAN_SIGLIP), centered(dino, MEAN_DINO)], WEIGHTS);
}

export function loadSpace(): { items: SpaceItem[]; clusters: SpaceCluster[] } {
  const items = raw.items.map((it) => {
    const siglip = decodeHalf(it.siglip);
    const dino = decodeHalf(it.dino);
    normalize(siglip);
    normalize(dino);
    return {
      id: it.id,
      thumb: it.thumb,
      w: it.w,
      h: it.h,
      siglip,
      fused: fusedVector(siglip, dino),
      hub: it.hub,
      textHub: it.textHub,
      pos: it.pos as Vec3,
      cluster: it.cluster,
      tags: it.tags,
    };
  });
  const clusters = raw.clusters.map((c) => ({
    label: c.label,
    level: c.level as 0 | 1,
    parent: c.parent,
    size: c.size,
    pos: c.pos as Vec3,
  }));
  return { items, clusters };
}

// Visual ranking against a fused query vector, best first.
export function rankByVector(query: Float32Array, items: SpaceItem[]): number[] {
  const score = items.map((it) => 2 * dot(query, it.fused) - it.hub);
  return items.map((_, i) => i).sort((a, b) => score[b] - score[a]);
}

// Text ranking: SigLIP2 text vector against the raw image vectors, minus half
// of each image's text hub. Without it, images that match words in general
// (screenshots full of text) rank high for every query.
export function rankByText(query: Float32Array, items: SpaceItem[]): number[] {
  const score = items.map((it) => dot(query, it.siglip) - 0.5 * it.textHub);
  return items.map((_, i) => i).sort((a, b) => score[b] - score[a]);
}

// A cluster as a query: its own images first (closest to the cluster's center
// first), then everything else by similarity to that center.
export function rankByCluster(cluster: number, items: SpaceItem[]): number[] {
  const members = items.filter((it) => it.cluster === cluster);
  if (members.length === 0) return items.map((_, i) => i);
  const center = new Float32Array(members[0].fused.length);
  for (const m of members) for (let d = 0; d < center.length; d++) center[d] += m.fused[d];
  normalize(center);
  const order = rankByVector(center, items);
  return [...order.filter((i) => items[i].cluster === cluster), ...order.filter((i) => items[i].cluster !== cluster)];
}

// Place an uploaded image in the space: hub penalty from its own neighbors,
// cluster by a similarity-weighted vote of its nearest images, position at the
// weighted mean of theirs (nudged so it does not sit exactly on one of them).
// The vocabulary is not in the browser, so its text hub is the board's average.
export function placeUpload(
  fused: Float32Array,
  items: SpaceItem[]
): { hub: number; textHub: number; cluster: number; pos: Vec3 } {
  const sims = items.map((it) => dot(fused, it.fused));
  const near = items.map((_, i) => i).sort((a, b) => sims[b] - sims[a]).slice(0, HUB_K);
  const hub = near.reduce((s, i) => s + sims[i], 0) / Math.max(1, near.length);
  const top = near.slice(0, 5);
  const votes = new Map<number, number>();
  const pos: Vec3 = [0, 0, 0];
  let wsum = 0;
  for (const i of top) {
    const w = Math.exp(sims[i] * 8);
    votes.set(items[i].cluster, (votes.get(items[i].cluster) ?? 0) + w);
    for (let d = 0; d < 3; d++) pos[d] += w * items[i].pos[d];
    wsum += w;
  }
  const cluster = [...votes.entries()].sort((a, b) => b[1] - a[1])[0]?.[0] ?? 0;
  const jitter = 0.03;
  for (let d = 0; d < 3; d++) pos[d] = pos[d] / Math.max(wsum, 1e-9) + (Math.random() - 0.5) * jitter;
  const textHub = items.reduce((s, it) => s + it.textHub, 0) / Math.max(1, items.length);
  return { hub, textHub, cluster, pos };
}
