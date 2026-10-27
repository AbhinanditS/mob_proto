// Vector helpers shared by the offline pipeline (scripts/precompute.ts) and the
// browser (src/app/space.ts). Pure functions, no DOM and no Node APIs.

export function dot(a: Float32Array, b: Float32Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

// In place. Returns the original norm.
export function normalize(v: Float32Array): number {
  let n = 0;
  for (let i = 0; i < v.length; i++) n += v[i] * v[i];
  n = Math.sqrt(n);
  if (n > 0) for (let i = 0; i < v.length; i++) v[i] /= n;
  return n;
}

// normalize(normalize(v) - mean). Removing the dataset mean takes out the
// component that every image shares, which otherwise dominates cosine
// similarity and turns a few images into "hubs" that match everything.
export function centered(v: Float32Array, mean: Float32Array): Float32Array {
  const out = Float32Array.from(v);
  normalize(out);
  for (let i = 0; i < out.length; i++) out[i] -= mean[i];
  normalize(out);
  return out;
}

// Concatenate per-model vectors, each scaled by sqrt(weight), so the dot
// product of two fused vectors is the weighted sum of per-model cosines.
export function fuse(parts: Float32Array[], weights: number[]): Float32Array {
  const total = weights.reduce((s, w) => s + w, 0);
  const len = parts.reduce((s, p) => s + p.length, 0);
  const out = new Float32Array(len);
  let o = 0;
  parts.forEach((p, k) => {
    const s = Math.sqrt(weights[k] / total);
    for (let i = 0; i < p.length; i++) out[o + i] = p[i] * s;
    o += p.length;
  });
  return out;
}

// ---- float16 storage ---------------------------------------------------------
// Vectors ship as base64 float16: half the size of float32, and cosine error
// stays below 1e-3.

const f32 = new Float32Array(1);
const u32 = new Uint32Array(f32.buffer);

function toHalf(value: number): number {
  f32[0] = value;
  const x = u32[0];
  const sign = (x >>> 16) & 0x8000;
  const exp = (x >>> 23) & 0xff;
  let mant = x & 0x7fffff;
  if (exp === 0xff) return sign | 0x7c00 | (mant ? 0x200 : 0);
  const e = exp - 127 + 15;
  if (e >= 0x1f) return sign | 0x7c00;
  if (e <= 0) {
    if (e < -10) return sign;
    mant |= 0x800000;
    const shift = 14 - e;
    let half = mant >> shift;
    if ((mant >> (shift - 1)) & 1) half += 1;
    return sign | half;
  }
  let half = sign | (e << 10) | (mant >> 13);
  if (mant & 0x1000) half += 1; // round to nearest
  return half;
}

function fromHalf(h: number): number {
  const sign = h & 0x8000 ? -1 : 1;
  const exp = (h >> 10) & 0x1f;
  const mant = h & 0x3ff;
  if (exp === 0) return sign * mant * 2 ** -24;
  if (exp === 0x1f) return mant ? NaN : sign * Infinity;
  return sign * (1 + mant / 1024) * 2 ** (exp - 15);
}

const B64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
const B64_INDEX = new Uint8Array(128);
for (let i = 0; i < B64.length; i++) B64_INDEX[B64.charCodeAt(i)] = i;

function bytesToBase64(bytes: Uint8Array): string {
  let out = "";
  for (let i = 0; i < bytes.length; i += 3) {
    const a = bytes[i];
    const b = i + 1 < bytes.length ? bytes[i + 1] : 0;
    const c = i + 2 < bytes.length ? bytes[i + 2] : 0;
    out += B64[a >> 2] + B64[((a & 3) << 4) | (b >> 4)];
    out += i + 1 < bytes.length ? B64[((b & 15) << 2) | (c >> 6)] : "=";
    out += i + 2 < bytes.length ? B64[c & 63] : "=";
  }
  return out;
}

function base64ToBytes(s: string): Uint8Array {
  const clean = s.replace(/=+$/, "");
  const out = new Uint8Array(Math.floor((clean.length * 3) / 4));
  let o = 0;
  for (let i = 0; i < clean.length; i += 4) {
    const a = B64_INDEX[clean.charCodeAt(i)];
    const b = B64_INDEX[clean.charCodeAt(i + 1)];
    const c = i + 2 < clean.length ? B64_INDEX[clean.charCodeAt(i + 2)] : 0;
    const d = i + 3 < clean.length ? B64_INDEX[clean.charCodeAt(i + 3)] : 0;
    out[o++] = (a << 2) | (b >> 4);
    if (o < out.length) out[o++] = ((b & 15) << 4) | (c >> 2);
    if (o < out.length) out[o++] = ((c & 3) << 6) | d;
  }
  return out;
}

export function encodeHalf(v: ArrayLike<number>): string {
  const bytes = new Uint8Array(v.length * 2);
  for (let i = 0; i < v.length; i++) {
    const h = toHalf(v[i]);
    bytes[i * 2] = h & 0xff; // little-endian
    bytes[i * 2 + 1] = h >> 8;
  }
  return bytesToBase64(bytes);
}

export function decodeHalf(s: string): Float32Array {
  const bytes = base64ToBytes(s);
  const out = new Float32Array(bytes.length >> 1);
  for (let i = 0; i < out.length; i++) out[i] = fromHalf(bytes[i * 2] | (bytes[i * 2 + 1] << 8));
  return out;
}
