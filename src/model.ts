// Checkpoints shared by every embedding consumer: scripts/precompute.ts
// (offline batch), text search, and the client-side upload pipeline. All of
// them must agree, because vectors from different checkpoints are not
// comparable.
//
// SIGLIP_MODEL carries meaning: its image vectors answer text queries and form
// half of the visual similarity. DINO_MODEL carries visual structure (layout,
// material, composition) and forms the other half.
export const SIGLIP_MODEL = "onnx-community/siglip2-base-patch16-384-ONNX";
export const DINO_MODEL = "onnx-community/dinov3-vitb16-pretrain-lvd1689m-ONNX";

// SigLIP2 was trained on lowercased captions padded to 64 tokens. Every text
// that goes through the text tower must use this exact preparation.
export const TEXT_MAX_LENGTH = 64;
export function prepareText(text: string): string {
  return text.trim().toLowerCase();
}

// Prompt ensemble. A text is embedded in every template and the normalized
// vectors are averaged, which smooths over the quirks of any single phrasing.
// Used for search queries and for the offline vocabulary.
export const TEXT_TEMPLATES = ["{}", "an image of {}", "{} style", "{}, design inspiration"];
