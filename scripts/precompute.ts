// Offline pipeline for the image space:
//
//   images → SigLIP2 + DINOv3 vectors → drop duplicates → fused similarity
//          → two-level clusters → 3D layout → zero-shot names → src/data/space.json
//
// Sources: originals in public/images when that folder has images (thumbnails
// are rebuilt from them). Otherwise the thumbnails listed by the current data
// file, so the space can be rebuilt from what the repository ships.
//
// Run: npm run precompute

import fs from "fs";
import path from "path";
import sharp from "sharp";
import {
  AutoModel,
  AutoProcessor,
  AutoTokenizer,
  RawImage,
  SiglipTextModel,
  SiglipVisionModel,
} from "@huggingface/transformers";
import { DINO_MODEL, SIGLIP_MODEL, TEXT_MAX_LENGTH, TEXT_TEMPLATES, prepareText } from "../src/model";
import { centered, decodeHalf, encodeHalf, fuse, normalize } from "../src/lib/vectors";
import { BROAD, SPECIFIC } from "./vocabulary";
import {
  affinity,
  buildHierarchy,
  cslsNeighbors,
  duplicateIndices,
  layout3D,
  localDensity,
  meanVector,
  nameClusters,
  similarityMatrix,
  tagScores,
  topTerms,
  type Vec,
} from "./space-build";

const ROOT = process.cwd();
const IMAGES_DIR = path.join(ROOT, "public", "images");
const THUMBS_DIR = path.join(ROOT, "public", "thumbs");
const OUT_FILE = path.join(ROOT, "src", "data", "space.json");
const LEGACY_FILE = path.join(ROOT, "src", "embeddings.json");
const CACHE_DIR = path.join(ROOT, ".cache", "precompute");

const WEIGHTS = { siglip: 0.5, dino: 0.5 };
const THUMB_SIZE = 768; // also the size every image is embedded from
const CSLS_K = 10;
const TAGS_PER_IMAGE = 3;
const TEXT_HUB_K = 10;
const LAYOUT = { neighbors: 15, minDist: 0.6, spread: 1.2, epochs: 600, clusterWeight: 0.3, seed: 1234567 };

interface Source {
  id: string;
  thumb: string;
  w: number;
  h: number;
  input: string; // file the vectors are computed from
}

const EXTENSIONS = new Set([".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp", ".gif", ".avif"]);

async function collectSources(): Promise<Source[]> {
  const originals = fs.existsSync(IMAGES_DIR)
    ? fs.readdirSync(IMAGES_DIR).filter((f) => EXTENSIONS.has(path.extname(f).toLowerCase())).sort()
    : [];
  if (originals.length > 0) {
    fs.mkdirSync(THUMBS_DIR, { recursive: true });
    const out: Source[] = [];
    for (const file of originals) {
      const input = path.join(IMAGES_DIR, file);
      const id = path.basename(file, path.extname(file)).replace(/[^a-zA-Z0-9-_]/g, "_");
      const meta = await sharp(input).metadata();
      if (!meta.width || !meta.height) {
        console.warn(`skip ${file}: no dimensions`);
        continue;
      }
      const rotated = (meta.orientation ?? 1) >= 5; // EXIF 5-8 swap width and height
      const thumb = `${id}.webp`;
      await sharp(input)
        .rotate()
        .resize({ width: THUMB_SIZE, height: THUMB_SIZE, fit: "inside", withoutEnlargement: true })
        .webp({ quality: 82 })
        .toFile(path.join(THUMBS_DIR, thumb));
      out.push({
        id,
        thumb: `/thumbs/${thumb}`,
        w: rotated ? meta.height : meta.width,
        h: rotated ? meta.width : meta.height,
        input,
      });
    }
    return out;
  }
  const manifestFile = fs.existsSync(OUT_FILE) ? OUT_FILE : LEGACY_FILE;
  if (!fs.existsSync(manifestFile)) return [];
  const raw = JSON.parse(fs.readFileSync(manifestFile, "utf8"));
  const list: { id: string; thumb: string; w: number; h: number }[] = Array.isArray(raw) ? raw : raw.items;
  console.log(`No originals in public/images; rebuilding from ${list.length} thumbnails.`);
  return list.map((it) => ({ id: it.id, thumb: it.thumb, w: it.w, h: it.h, input: path.join(ROOT, "public", it.thumb) }));
}

// ---- cache: vectors keyed by source file and model, so re-runs only embed what changed

function loadCache(name: string): Record<string, string> {
  const file = path.join(CACHE_DIR, `${name}.json`);
  return fs.existsSync(file) ? JSON.parse(fs.readFileSync(file, "utf8")) : {};
}

function saveCache(name: string, data: Record<string, string>) {
  fs.mkdirSync(CACHE_DIR, { recursive: true });
  fs.writeFileSync(path.join(CACHE_DIR, `${name}.json`), JSON.stringify(data));
}

// Every image is embedded from the same 768px rendition the browser shows, with
// transparency flattened onto white, so originals and thumbnails give the same
// vectors.
async function loadImage(file: string): Promise<RawImage> {
  const { data, info } = await sharp(file)
    .rotate()
    .resize({ width: THUMB_SIZE, height: THUMB_SIZE, fit: "inside", withoutEnlargement: true })
    .flatten({ background: "#ffffff" })
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  return new RawImage(new Uint8ClampedArray(data), info.width, info.height, 3);
}

async function embedImages(sources: Source[]): Promise<{ siglip: Vec[]; dino: Vec[] }> {
  const cache = loadCache("images");
  const keys = sources.map((s) => {
    const st = fs.statSync(s.input);
    return `${SIGLIP_MODEL}|${DINO_MODEL}|${s.input}|${st.size}|${st.mtimeMs}`;
  });
  const missing = sources.map((_, i) => i).filter((i) => !cache[keys[i]]);
  if (missing.length > 0) {
    console.log(`Embedding ${missing.length} images (SigLIP2 + DINOv3)…`);
    const sigProcessor = await AutoProcessor.from_pretrained(SIGLIP_MODEL);
    const sigModel = await SiglipVisionModel.from_pretrained(SIGLIP_MODEL, { dtype: "fp32" });
    const dinoProcessor = await AutoProcessor.from_pretrained(DINO_MODEL);
    const dinoModel = await AutoModel.from_pretrained(DINO_MODEL, { dtype: "fp32" });
    for (const [k, i] of missing.entries()) {
      try {
        const image = await loadImage(sources[i].input);
        const sig = (await sigModel(await sigProcessor(image))).pooler_output.data as Float32Array;
        const dino = (await dinoModel(await dinoProcessor(image))).pooler_output.data as Float32Array;
        cache[keys[i]] = `${encodeHalf(sig)}|${encodeHalf(dino)}`;
      } catch (err) {
        console.error(`Failed to embed ${sources[i].input}:`, err);
      }
      if ((k + 1) % 20 === 0 || k === missing.length - 1) {
        console.log(`  ${k + 1}/${missing.length}`);
        saveCache("images", cache);
      }
    }
  }
  const siglip: Vec[] = [];
  const dino: Vec[] = [];
  keys.forEach((key) => {
    const [a, b] = (cache[key] ?? "|").split("|");
    siglip.push(a ? decodeHalf(a) : new Float32Array(0));
    dino.push(b ? decodeHalf(b) : new Float32Array(0));
  });
  return { siglip, dino };
}

// One vector per term: the normalized mean over the prompt templates.
async function embedTerms(terms: string[]): Promise<Vec[]> {
  const cache = loadCache("text");
  const prompts = terms.flatMap((t) => TEXT_TEMPLATES.map((tpl) => prepareText(tpl.replace("{}", t))));
  const missing = [...new Set(prompts.filter((p) => !cache[`${SIGLIP_MODEL}|${p}`]))];
  if (missing.length > 0) {
    console.log(`Embedding ${missing.length} vocabulary prompts…`);
    const tokenizer = await AutoTokenizer.from_pretrained(SIGLIP_MODEL);
    const model = await SiglipTextModel.from_pretrained(SIGLIP_MODEL, { dtype: "q8" });
    for (const p of missing) {
      const inputs = tokenizer(p, { padding: "max_length", truncation: true, max_length: TEXT_MAX_LENGTH });
      const v = Float32Array.from((await model(inputs)).pooler_output.data as Float32Array);
      cache[`${SIGLIP_MODEL}|${p}`] = encodeHalf(v);
    }
    saveCache("text", cache);
  }
  return terms.map((t) => {
    const acc = new Float32Array(decodeHalf(cache[`${SIGLIP_MODEL}|${prepareText(TEXT_TEMPLATES[0].replace("{}", t))}`]).length);
    for (const tpl of TEXT_TEMPLATES) {
      const v = decodeHalf(cache[`${SIGLIP_MODEL}|${prepareText(tpl.replace("{}", t))}`]);
      normalize(v);
      for (let i = 0; i < acc.length; i++) acc[i] += v[i];
    }
    normalize(acc);
    return acc;
  });
}

const round = (x: number) => Math.round(x * 1e4) / 1e4;

async function main() {
  const all = await collectSources();
  if (all.length === 0) {
    console.log("No images found. Add images to public/images and run again.");
    return;
  }
  const embedded = await embedImages(all);
  const ok = all.map((_, i) => i).filter((i) => embedded.siglip[i].length > 0 && embedded.dino[i].length > 0);
  for (const v of [...embedded.siglip, ...embedded.dino]) normalize(v);

  const dupes = duplicateIndices(ok.map((i) => embedded.siglip[i]), ok.map((i) => embedded.dino[i]));
  const kept = ok.filter((_, k) => !dupes.has(k));
  console.log(`${all.length} images, ${all.length - ok.length} failed, ${dupes.size} exact duplicates dropped, ${kept.length} kept.`);
  const sources = kept.map((i) => all[i]);
  const siglip = kept.map((i) => embedded.siglip[i]);
  const dino = kept.map((i) => embedded.dino[i]);
  const n = sources.length;

  // Fused space: each model centered on this board's mean, weighted 1:1.
  const meanSiglip = meanVector(siglip);
  const meanDino = meanVector(dino);
  const fused = sources.map((_, i) =>
    fuse([centered(siglip[i], meanSiglip), centered(dino[i], meanDino)], [WEIGHTS.siglip, WEIGHTS.dino])
  );
  const S = similarityMatrix(fused);
  const r = localDensity(S, n, CSLS_K);

  const kFine = Math.max(2, Math.min(60, Math.round(n / 8.5), n));
  const kCoarse = Math.max(1, Math.min(12, Math.round(Math.sqrt(n) / 2), kFine - 1));
  const hierarchy = buildHierarchy(fused, S, localDensity(S, n, 5), kCoarse, kFine, 0.85);
  const neighbors = cslsNeighbors(S, n, r, LAYOUT.neighbors * 3); // spare candidates for layout3D
  console.log("Laying out the cloud…");
  const positions = layout3D(fused, S, neighbors, hierarchy.fine, LAYOUT);

  console.log("Naming clusters…");
  const specificTerms = await embedTerms(SPECIFIC);
  const broadTerms = await embedTerms(BROAD);
  const specific = affinity(siglip, specificTerms);
  const broad = affinity(siglip, broadTerms);
  const naming = nameClusters(hierarchy, specific, broad, specificTerms, broadTerms);
  const tags = tagScores(specific).map((s) => topTerms(s, TAGS_PER_IMAGE).map((t) => SPECIFIC[t]));
  // Text hub: how strongly an image matches words in general. Search subtracts
  // it, so images that match every query do not crowd the top of every result.
  const textHub = sources.map((_, i) => {
    const best = [...specific[i], ...broad[i]].sort((a, b) => b - a);
    return best.slice(0, TEXT_HUB_K).reduce((s, x) => s + x, 0) / TEXT_HUB_K;
  });

  const centroid = (members: number[]) =>
    [0, 1, 2].map((d) => round(members.reduce((s, i) => s + positions[i][d], 0) / members.length));
  const clusters = [
    ...naming.coarse.map((t, c) => {
      const members = sources.map((_, i) => i).filter((i) => hierarchy.coarse[i] === c);
      return { label: BROAD[t], level: 0, parent: null as number | null, size: members.length, pos: centroid(members) };
    }),
    ...naming.fine.map((t, f) => {
      const members = sources.map((_, i) => i).filter((i) => hierarchy.fine[i] === f);
      const core = members.filter((i) => hierarchy.core[i]);
      return {
        label: SPECIFIC[t],
        level: 1,
        parent: hierarchy.parentOf[f] as number | null,
        size: members.length,
        pos: centroid(core.length > 0 ? core : members),
      };
    }),
  ];

  const out = {
    version: 2,
    models: { siglip: SIGLIP_MODEL, dino: DINO_MODEL },
    weights: WEIGHTS,
    means: { siglip: encodeHalf(meanSiglip), dino: encodeHalf(meanDino) },
    clusters,
    items: sources.map((s, i) => ({
      id: s.id,
      thumb: s.thumb,
      w: s.w,
      h: s.h,
      cluster: kCoarse + hierarchy.fine[i],
      tags: tags[i],
      hub: round(r[i]),
      textHub: round(textHub[i]),
      pos: positions[i].map(round),
      siglip: encodeHalf(siglip[i]),
      dino: encodeHalf(dino[i]),
    })),
  };
  fs.mkdirSync(path.dirname(OUT_FILE), { recursive: true });
  fs.writeFileSync(OUT_FILE, JSON.stringify(out));
  console.log(`Wrote ${OUT_FILE} (${(fs.statSync(OUT_FILE).size / 1e6).toFixed(2)} MB)`);
  for (let c = 0; c < kCoarse; c++) {
    console.log(`  ${clusters[c].label.toUpperCase()} (${clusters[c].size})`);
    for (const cl of clusters) if (cl.level === 1 && cl.parent === c) console.log(`    ${cl.label} (${cl.size})`);
  }
}

main().catch((err) => {
  console.error("Precompute failed:", err);
  process.exit(1);
});
