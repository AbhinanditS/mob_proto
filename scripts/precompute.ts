import { AutoProcessor, SiglipVisionModel, RawImage } from '@huggingface/transformers';
import sharp from 'sharp';
import fs from 'fs';
import path from 'path';

const IMAGES_DIR = path.join(process.cwd(), 'public', 'images');
const THUMBS_DIR = path.join(process.cwd(), 'public', 'thumbs');
const OUTPUT_FILE = path.join(process.cwd(), 'src', 'embeddings.json');

const MODEL = 'Xenova/siglip-base-patch16-384';

async function main() {
  console.log('Loading SigLIP model and processor...');
  const processor = await AutoProcessor.from_pretrained(MODEL);
  const vision = await SiglipVisionModel.from_pretrained(MODEL, { dtype: 'fp32' });
  console.log('Model loaded successfully.');

  // Ensure output directories exist
  if (!fs.existsSync(THUMBS_DIR)) {
    fs.mkdirSync(THUMBS_DIR, { recursive: true });
  }
  const srcDir = path.dirname(OUTPUT_FILE);
  if (!fs.existsSync(srcDir)) {
    fs.mkdirSync(srcDir, { recursive: true });
  }

  // Scan public/images
  if (!fs.existsSync(IMAGES_DIR)) {
    console.error(`Images directory does not exist at ${IMAGES_DIR}. Please create it and add your images.`);
    process.exit(1);
  }

  const files = fs.readdirSync(IMAGES_DIR);
  const supportedExtensions = new Set(['.png', '.jpg', '.jpeg', '.webp', '.tiff', '.bmp', '.gif']);
  const imageFiles = files.filter(f => supportedExtensions.has(path.extname(f).toLowerCase()));

  if (imageFiles.length === 0) {
    console.log(`No images found in ${IMAGES_DIR}. Please drop your images there and run this script again.`);
    process.exit(0);
  }

  console.log(`Found ${imageFiles.length} image files to process.`);

  const results = [];

  for (let i = 0; i < imageFiles.length; i++) {
    const filename = imageFiles[i];
    const imagePath = path.join(IMAGES_DIR, filename);
    const id = path.basename(filename, path.extname(filename))
      .replace(/[^a-zA-Z0-9-_]/g, '_');

    console.log(`[${i + 1}/${imageFiles.length}] Processing ${filename} (ID: ${id})...`);

    try {
      // 1. Get metadata and dimensions
      const metadata = await sharp(imagePath).metadata();
      const { width, height } = metadata;
      if (!width || !height) {
        throw new Error('Could not retrieve image dimensions');
      }

      // 2. Generate thumbnail
      const thumbFilename = `${id}.webp`;
      const thumbPath = path.join(THUMBS_DIR, thumbFilename);
      await sharp(imagePath)
        .resize({ width: 768, height: 768, fit: 'inside', withoutEnlargement: true })
        .webp({ quality: 82 })
        .toFile(thumbPath);

      // 3. Compute embedding
      const rawImage = await RawImage.read(imagePath);
      const inputs = await processor(rawImage);
      const { pooler_output } = await vision(inputs);
      const v = Array.from(pooler_output.data as Float32Array);
      const norm = Math.hypot(...v);
      const embedding = v.map(x => Math.round((x / norm) * 1e6) / 1e6);

      results.push({
        id,
        file: `/images/${filename}`,
        thumb: `/thumbs/${thumbFilename}`,
        w: width,
        h: height,
        embedding
      });
    } catch (err) {
      console.error(`Failed to process ${filename}:`, err);
    }
  }

  console.log(`Successfully processed ${results.length}/${imageFiles.length} images.`);
  fs.writeFileSync(OUTPUT_FILE, JSON.stringify(results, null, 2), 'utf-8');
  console.log(`Saved embeddings to ${OUTPUT_FILE}`);
}

main().catch(err => {
  console.error('Fatal error in precompute script:', err);
  process.exit(1);
});
