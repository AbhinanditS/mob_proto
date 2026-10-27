// Browser-side embedding for text search and uploads. Models load lazily on
// first use (transformers.js caches the downloads) and use the same checkpoints
// as scripts/precompute.ts, so their vectors are comparable with the dataset.

import type { PreTrainedModel, PreTrainedTokenizer, Processor } from "@huggingface/transformers";
import { DINO_MODEL, SIGLIP_MODEL, TEXT_MAX_LENGTH, TEXT_TEMPLATES, prepareText } from "@/model";
import { normalize } from "@/lib/vectors";

// Uploads are embedded from the same 768px rendition (on white) that the
// offline pipeline uses.
const EMBED_SIZE = 768;

let textModel: Promise<{ tokenizer: PreTrainedTokenizer; model: PreTrainedModel }> | null = null;
let visionModels: Promise<{
  sigProcessor: Processor;
  sigModel: PreTrainedModel;
  dinoProcessor: Processor;
  dinoModel: PreTrainedModel;
}> | null = null;

async function transformers() {
  const t = await import("@huggingface/transformers");
  t.env.allowLocalModels = false;
  return t;
}

// The in-flight promise is shared, so concurrent callers trigger one download.
// A failed load clears it, so the next call retries.
function loadTextModel() {
  textModel ??= (async () => {
    const { AutoTokenizer, SiglipTextModel } = await transformers();
    const tokenizer = await AutoTokenizer.from_pretrained(SIGLIP_MODEL);
    const model = await SiglipTextModel.from_pretrained(SIGLIP_MODEL, { dtype: "q8" });
    return { tokenizer, model };
  })().catch((err) => {
    textModel = null;
    throw err;
  });
  return textModel;
}

function loadVisionModels() {
  visionModels ??= (async () => {
    const { AutoModel, AutoProcessor, SiglipVisionModel } = await transformers();
    const [sigProcessor, sigModel, dinoProcessor, dinoModel] = await Promise.all([
      AutoProcessor.from_pretrained(SIGLIP_MODEL),
      SiglipVisionModel.from_pretrained(SIGLIP_MODEL, { dtype: "q8" }),
      AutoProcessor.from_pretrained(DINO_MODEL),
      AutoModel.from_pretrained(DINO_MODEL, { dtype: "q8" }),
    ]);
    return { sigProcessor, sigModel, dinoProcessor, dinoModel };
  })().catch((err) => {
    visionModels = null;
    throw err;
  });
  return visionModels;
}

export function isTextModelLoaded(): boolean {
  return textModel !== null;
}

// One query vector: the normalized mean over the prompt templates, the same
// way the offline vocabulary is embedded.
export async function embedText(text: string): Promise<Float32Array> {
  const { tokenizer, model } = await loadTextModel();
  const prompts = TEXT_TEMPLATES.map((t) => prepareText(t.replace("{}", text.trim())));
  const inputs = tokenizer(prompts, { padding: "max_length", truncation: true, max_length: TEXT_MAX_LENGTH });
  const { pooler_output } = await model(inputs);
  const data = pooler_output.data as Float32Array;
  const dim = data.length / prompts.length;
  const out = new Float32Array(dim);
  for (let p = 0; p < prompts.length; p++) {
    const v = data.slice(p * dim, (p + 1) * dim);
    normalize(v);
    for (let i = 0; i < dim; i++) out[i] += v[i];
  }
  normalize(out);
  return out;
}

export async function embedImage(source: Blob): Promise<{ siglip: Float32Array; dino: Float32Array }> {
  const [{ RawImage }, models] = await Promise.all([transformers(), loadVisionModels()]);
  const bitmap = await createImageBitmap(source);
  const scale = Math.min(1, EMBED_SIZE / Math.max(bitmap.width, bitmap.height));
  const canvas = document.createElement("canvas");
  canvas.width = Math.max(1, Math.round(bitmap.width * scale));
  canvas.height = Math.max(1, Math.round(bitmap.height * scale));
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) throw new Error("Canvas 2D context unavailable");
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
  bitmap.close();
  const image = RawImage.fromCanvas(canvas);
  const sigOut = await models.sigModel(await models.sigProcessor(image));
  const dinoOut = await models.dinoModel(await models.dinoProcessor(image));
  const siglip = Float32Array.from(sigOut.pooler_output.data as Float32Array);
  const dino = Float32Array.from(dinoOut.pooler_output.data as Float32Array);
  normalize(siglip);
  normalize(dino);
  return { siglip, dino };
}
