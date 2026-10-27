# River

River shows an image board by visual similarity and by aesthetic.

- In the grid, click an image to see the images most like it. The closest matches come first and are the largest. You can search with words, or add your own images and videos for the session.
- The cloud shows the board as a 3D cloud of cluster names. Move the cursor over the cloud to see the images under it: the nearest image is full size, and images further away are smaller. Click an image to open its matches in the grid. Drag to rotate and scroll to zoom.

## How similarity works

Each image has two vectors: SigLIP2 base (meaning, also used for text search) and DINOv3 base (visual structure such as layout, material and composition). Each vector is centered on the board's mean, and the two are combined with equal weight. Ranking subtracts a hub penalty (CSLS), so a few average-looking images cannot appear near the top of every result. Exact duplicate pins are removed.

Text search embeds the query in four prompt templates and averages them. It also subtracts a per-image text-hub value, so screenshots full of text do not match every query.

Clusters are two levels of one Ward tree over the combined vectors. Names come from a curated vocabulary (`scripts/vocabulary.ts`), scored zero-shot with the SigLIP2 text model.

## Run

```bash
npm install
npm run dev
```

## Rebuild the data

Put the original images in `public/images`, then run:

```bash
npm run precompute
```

The script makes the thumbnails, the vectors, the 3D layout, the clusters and their names, and writes `src/data/space.json`. Without `public/images`, it rebuilds from the thumbnails that `src/data/space.json` lists. The first run downloads about 1 GB of models. Results are cached in `.cache/precompute`.

In the browser, the first text search downloads the SigLIP2 text model (about 280 MB). The first upload downloads the two vision models (about 180 MB). Both stay cached.
