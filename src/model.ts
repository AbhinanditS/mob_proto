// SigLIP checkpoint shared by every embedding consumer: scripts/precompute.ts
// (offline batch), page.tsx's text search, and the client-side upload
// pipeline. All three must agree — vectors from different checkpoints are not
// comparable via dot().
export const MODEL = "Xenova/siglip-large-patch16-384";
