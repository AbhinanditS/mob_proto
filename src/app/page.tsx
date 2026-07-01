"use client";

import React, { useState, useEffect, useRef, useMemo } from "react";
import type { PreTrainedTokenizer, PreTrainedModel, Processor, RawImage } from "@huggingface/transformers";
import rawEmbeddings from "@/embeddings.json";
import { MODEL } from "@/model";
import Converge from "./Converge";
import Navigator3D from "./Navigator3D";

interface ImageRecord {
  id: string;
  file: string;
  thumb: string;
  w: number;
  h: number;
  embedding: number[];
  kind?: "image" | "video"; // undefined/omitted == "image" for the precomputed dataset
}

interface LaidOutItem extends ImageRecord {
  left: number;
  top: number;
  width: number;
  height: number;
}

const VIDEO_POSTER_SEEK_FRACTION = 0.1; // seek 10% into the clip — frame 0 is often black/blank

// Extract one representative frame from a video File via an offscreen
// <video>+<canvas> — no ffmpeg, no server round-trip. The frame is embedded
// exactly like an uploaded image; the original video stays playable in the
// grid via its own object URL.
async function extractVideoPosterFrame(
  file: File
): Promise<{ videoUrl: string; posterBlob: Blob; w: number; h: number }> {
  const videoUrl = URL.createObjectURL(file);
  const video = document.createElement("video");
  video.src = videoUrl;
  video.muted = true;
  video.playsInline = true;

  await new Promise<void>((resolve, reject) => {
    video.onloadedmetadata = () => resolve();
    video.onerror = () => reject(new Error("Failed to load video metadata"));
  });

  const seekTime = Math.min(video.duration * VIDEO_POSTER_SEEK_FRACTION, video.duration || 0);
  await new Promise<void>((resolve, reject) => {
    video.onseeked = () => resolve();
    video.onerror = () => reject(new Error("Failed to seek video"));
    video.currentTime = seekTime;
  });

  const w = video.videoWidth;
  const h = video.videoHeight;
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Canvas 2D context unavailable");
  ctx.drawImage(video, 0, 0, w, h);

  const posterBlob = await new Promise<Blob>((resolve, reject) => {
    canvas.toBlob(
      (blob) => (blob ? resolve(blob) : reject(new Error("Canvas toBlob failed"))),
      "image/webp",
      0.9
    );
  });

  return { videoUrl, posterBlob, w, h };
}

// Precomputed dataset IDs are always "NNN_..." (filename-derived, see
// scripts/precompute.ts) — this prefix can never collide with them.
let uploadIdSeq = 0;
function nextUploadId(): string {
  uploadIdSeq += 1;
  return `upload-${Date.now()}-${uploadIdSeq}`;
}

export default function Home() {
  const [mounted, setMounted] = useState(false);
  const [queryId, setQueryId] = useState<string | null>(null);
  const [history, setHistory] = useState<string[]>([]);
  const [showHelp, setShowHelp] = useState(false);
  const [dimensions, setDimensions] = useState({ w: 1200, h: 800 });
  // Converge: a self-contained pairwise-choice overlay. It reads the same
  // embeddings but never touches queryId/textVec/neighborhood, so the main
  // grid's geometry is frozen while it runs.
  const [convergeActive, setConvergeActive] = useState(false);
  // 3D navigator: a self-contained WebGL overlay over the same embeddings. Like
  // Converge, it never touches the main grid's queryId/textVec/neighborhood.
  const [navigatorActive, setNavigatorActive] = useState(false);

  // Text content-search state
  const [searchTerm, setSearchTerm] = useState("");
  const [textVec, setTextVec] = useState<Float32Array | null>(null);
  const [searchStatus, setSearchStatus] = useState<string | null>(null);

  // Upload state: files the user drops/picks are embedded client-side and
  // merged into `items` for this session only — never written to disk, gone
  // on refresh.
  const [uploadStatus, setUploadStatus] = useState<string | null>(null);
  const [isDraggingFile, setIsDraggingFile] = useState(false);

  const helpRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  // Cache the lazily-loaded tokenizer + text model so only the first search
  // pays the load/download cost.
  const textModelRef = useRef<{ tokenizer: PreTrainedTokenizer; model: PreTrainedModel } | null>(null);
  // Same lazy-load-once pattern for embedding uploaded images/video frames.
  const visionModelRef = useRef<{ processor: Processor; model: PreTrainedModel } | null>(null);
  // Monotonic token to invalidate in-flight searches when the user clears the
  // box, submits a new query, or clicks away before the previous one resolves.
  const searchSeqRef = useRef(0);
  // Same pattern for uploads: dropping a new batch while a previous one is
  // still embedding invalidates the stale batch's in-flight loop.
  const uploadSeqRef = useRef(0);
  // Counts nested dragenter/dragleave (fired on every child element under the
  // cursor) so the drop-zone overlay only toggles on the outermost enter/leave.
  const dragCounterRef = useRef(0);

  // Items: the precomputed dataset, plus anything dropped in this session.
  // Lazy initializer preserves the old useMemo's "compute once" semantics for
  // the static import, while making the array appendable via setItems.
  const [items, setItems] = useState<(ImageRecord & { vec: Float32Array })[]>(() =>
    (rawEmbeddings as ImageRecord[]).map(item => ({
      ...item,
      vec: new Float32Array(item.embedding),
    }))
  );

  // Map for fast O(1) retrieval by ID
  const byId = useMemo(() => {
    return new Map(items.map(item => [item.id, item]));
  }, [items]);

  // Set mounted and dimensions client-side, initialize seed query
  useEffect(() => {
    setMounted(true);

    const handleResize = () => {
      setDimensions({
        w: window.innerWidth,
        h: window.innerHeight
      });
    };

    window.addEventListener("resize", handleResize);
    handleResize();

    // Determine initial seed image
    if (items.length > 0) {
      let initialId = "";
      const hash = window.location.hash.substring(1);
      if (hash && byId.has(hash)) {
        initialId = hash;
      } else {
        // Pick a random image to seed the experience
        const randomIndex = Math.floor(Math.random() * items.length);
        initialId = items[randomIndex].id;
      }
      setQueryId(initialId);
    }

    return () => {
      window.removeEventListener("resize", handleResize);
    };
  }, [items, byId]);

  // Listen to hashchange events (native browser back/forward support)
  useEffect(() => {
    if (!mounted) return;

    const handleHashChange = () => {
      const hash = window.location.hash.substring(1);
      if (hash && byId.has(hash) && hash !== queryId) {
        setQueryId(hash);
      }
    };

    window.addEventListener("hashchange", handleHashChange);
    return () => {
      window.removeEventListener("hashchange", handleHashChange);
    };
  }, [mounted, queryId, byId]);

  // Close help overlay if clicked outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (showHelp && helpRef.current && !helpRef.current.contains(event.target as Node)) {
        setShowHelp(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [showHelp]);

  // Brute-force Cosine Similarity (Dot Product over pre-normalized vectors)
  const dot = (a: Float32Array, b: Float32Array) => {
    let s = 0;
    const len = a.length;
    for (let i = 0; i < len; i++) s += a[i] * b[i];
    return s;
  };

  // Neighborhood: rank all items against an effective query vector. The query
  // is either a typed text embedding (content search) or, falling back, the
  // clicked image's own vector (visual exploration).
  const neighborhood = useMemo(() => {
    if (items.length === 0) return [];

    let q: Float32Array | null = textVec;
    if (!q) {
      if (!queryId) return [];
      const queryItem = byId.get(queryId);
      if (!queryItem) return [];
      q = queryItem.vec;
    }

    const query = q;
    return items
      .map(it => ({ it, s: dot(query, it.vec) }))
      .sort((a, b) => b.s - a.s)
      .map(x => x.it);
  }, [queryId, textVec, items, byId]);

  // Layout geometry computation (Justified Grid)
  const flatLaidOutItems = useMemo(() => {
    if (neighborhood.length === 0) return [];

    const containerW = dimensions.w;
    const containerH = dimensions.h;
    const targetH = Math.max(180, Math.min(320, containerH / 4.5));
    const gap = 1;

    const laidOut: LaidOutItem[] = [];
    let currentTop = 0;
    let currentRow: ImageRecord[] = [];
    let currentRowWidth = 0;

    for (let i = 0; i < neighborhood.length; i++) {
      const item = neighborhood[i];
      const aspect = item.w / item.h;
      const itemW = aspect * targetH;
      const additionalW = currentRow.length > 0 ? gap + itemW : itemW;

      if (currentRowWidth + additionalW > containerW && currentRow.length > 0) {
        // Committing this row: scale it down to exactly fit containerW
        const numGaps = currentRow.length - 1;
        const availableW = containerW - numGaps * gap;
        const sumAspect = currentRow.reduce((sum, it) => sum + (it.w / it.h), 0);
        const rowH = availableW / sumAspect;

        let currentLeft = 0;
        for (const rowItem of currentRow) {
          const finalW = (rowItem.w / rowItem.h) * rowH;
          laidOut.push({
            ...rowItem,
            left: currentLeft,
            top: currentTop,
            width: finalW,
            height: rowH,
          });
          currentLeft += finalW + gap;
        }

        currentTop += rowH + gap;
        currentRow = [item];
        currentRowWidth = itemW;
      } else {
        currentRow.push(item);
        currentRowWidth += additionalW;
      }
    }

    // Justify the last row if there are remaining items
    if (currentRow.length > 0) {
      const numGaps = currentRow.length - 1;
      const availableW = containerW - numGaps * gap;
      const sumAspect = currentRow.reduce((sum, it) => sum + (it.w / it.h), 0);
      const rowH = availableW / sumAspect;

      let currentLeft = 0;
      for (const rowItem of currentRow) {
        const finalW = (rowItem.w / rowItem.h) * rowH;
        laidOut.push({
          ...rowItem,
          left: currentLeft,
          top: currentTop,
          width: finalW,
          height: rowH,
        });
        currentLeft += finalW + gap;
      }
    }

    return laidOut;
  }, [neighborhood, dimensions]);

  // Left click (hop to neighbor). Clicking a result hands control back to
  // image-to-image visual exploration, exiting text-search mode.
  const handleImageClick = (id: string) => {
    if (textVec || searchStatus) {
      searchSeqRef.current++;
      setTextVec(null);
      setSearchTerm("");
      setSearchStatus(null);
    }
    if (queryId !== id) {
      if (queryId) setHistory(prev => [...prev, queryId]);
      setQueryId(id);
      window.location.hash = id;
    }
  };

  // Embed a typed query with SigLIP's text tower (client-side, lazy) and use
  // it to drive the ranking. The image vectors already live in this same space.
  const runTextSearch = async (text: string) => {
    const query = text.trim();
    // Each call gets a token; later calls (or clearing) bump it, so a slow
    // request that resolves after the user has moved on becomes a no-op.
    const seq = ++searchSeqRef.current;
    if (!query) {
      setTextVec(null);
      setSearchStatus(null);
      return;
    }

    try {
      let cached = textModelRef.current;
      if (!cached) {
        setSearchStatus("Loading model…");
        const { AutoTokenizer, SiglipTextModel, env } = await import("@huggingface/transformers");
        // Fetch from the HuggingFace CDN; skip spurious local-path lookups.
        env.allowLocalModels = false;
        const tokenizer = await AutoTokenizer.from_pretrained(MODEL);
        const model = await SiglipTextModel.from_pretrained(MODEL, { dtype: "q8" });
        cached = { tokenizer, model };
        textModelRef.current = cached;
      }
      if (searchSeqRef.current !== seq) return;

      setSearchStatus("Searching…");
      // SigLIP expects fixed-length, padded (max_length 64) token sequences.
      const inputs = cached.tokenizer(query, { padding: "max_length", truncation: true });
      const { pooler_output } = await cached.model(inputs);
      if (searchSeqRef.current !== seq) return;

      const v = Array.from(pooler_output.data as Float32Array);
      // L2-normalize so dot product equals cosine similarity, matching how the
      // image vectors are normalized in scripts/precompute.ts.
      const norm = Math.hypot(...v) || 1;
      const vec = new Float32Array(v.map(x => x / norm));

      setTextVec(vec);
      setSearchStatus(null);
    } catch (err) {
      console.error("Text search failed:", err);
      if (searchSeqRef.current === seq) setSearchStatus("Search failed");
    }
  };

  const handleSearchKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      e.preventDefault();
      runTextSearch(searchTerm);
    }
  };

  const clearSearch = () => {
    searchSeqRef.current++;
    setSearchTerm("");
    setTextVec(null);
    setSearchStatus(null);
  };

  // Embed one already-loaded RawImage with SigLIP's vision tower (client-side,
  // lazy — mirrors runTextSearch's load-once-and-cache pattern above).
  const embedRawImage = async (raw: RawImage): Promise<Float32Array> => {
    let cached = visionModelRef.current;
    if (!cached) {
      const { AutoProcessor, SiglipVisionModel, env } = await import("@huggingface/transformers");
      env.allowLocalModels = false;
      const processor = await AutoProcessor.from_pretrained(MODEL);
      const model = await SiglipVisionModel.from_pretrained(MODEL, { dtype: "q8" });
      cached = { processor, model };
      visionModelRef.current = cached;
    }
    const inputs = await cached.processor(raw);
    const { pooler_output } = await cached.model(inputs);
    const v = Array.from(pooler_output.data as Float32Array);
    const norm = Math.hypot(...v) || 1;
    return new Float32Array(v.map(x => x / norm));
  };

  // Embed dropped/picked files client-side and merge them into `items`.
  // Images embed directly; videos get a representative frame extracted first
  // (extractVideoPosterFrame) and that frame is embedded the same way. Files
  // are processed sequentially, not concurrently — ONNX Runtime Web inference
  // is CPU-bound and largely serializes anyway, and sequential keeps the
  // done/total progress readout exact without a concurrency-pool utility.
  const ingestFiles = async (files: File[]) => {
    const seq = ++uploadSeqRef.current;
    const imageFiles = files.filter(f => f.type.startsWith("image/"));
    const videoFiles = files.filter(f => f.type.startsWith("video/"));
    const total = imageFiles.length + videoFiles.length;
    if (total === 0) return;

    const { RawImage } = await import("@huggingface/transformers");
    let done = 0;
    setUploadStatus(`Embedding 0 / ${total}…`);
    const newRecords: (ImageRecord & { vec: Float32Array })[] = [];

    try {
      for (const file of imageFiles) {
        if (uploadSeqRef.current !== seq) return;
        const objectUrl = URL.createObjectURL(file);
        const raw = await RawImage.fromURL(objectUrl);
        const vec = await embedRawImage(raw);
        newRecords.push({
          id: nextUploadId(),
          file: objectUrl,
          thumb: objectUrl,
          w: raw.width,
          h: raw.height,
          embedding: Array.from(vec),
          vec,
          kind: "image",
        });
        done++;
        setUploadStatus(`Embedding ${done} / ${total}…`);
      }

      for (const file of videoFiles) {
        if (uploadSeqRef.current !== seq) return;
        const { videoUrl, posterBlob, w, h } = await extractVideoPosterFrame(file);
        const posterUrl = URL.createObjectURL(posterBlob);
        const raw = await RawImage.fromBlob(posterBlob);
        const vec = await embedRawImage(raw);
        newRecords.push({
          id: nextUploadId(),
          file: videoUrl,
          thumb: posterUrl,
          w,
          h,
          embedding: Array.from(vec),
          vec,
          kind: "video",
        });
        done++;
        setUploadStatus(`Embedding ${done} / ${total}…`);
      }

      if (uploadSeqRef.current !== seq) return;
      setItems(prev => [...prev, ...newRecords]);
      setUploadStatus(null);
    } catch (err) {
      console.error("Upload embedding failed:", err);
      if (uploadSeqRef.current === seq) setUploadStatus("Upload failed");
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };
  const handleDragEnter = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.types.includes("Files")) {
      dragCounterRef.current += 1;
      setIsDraggingFile(true);
    }
  };
  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    dragCounterRef.current -= 1;
    if (dragCounterRef.current <= 0) {
      dragCounterRef.current = 0;
      setIsDraggingFile(false);
    }
  };
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    dragCounterRef.current = 0;
    setIsDraggingFile(false);
    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) ingestFiles(files);
  };
  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files ?? []);
    if (files.length > 0) ingestFiles(files);
    e.target.value = "";
  };

  // Right click (open original image in new tab)
  const handleImageRightClick = (e: React.MouseEvent, fileUrl: string) => {
    e.preventDefault();
    window.open(fileUrl, "_blank");
  };

  // Navigation handlers
  const handleBack = () => {
    if (history.length > 0) {
      const nextHistory = [...history];
      const prevId = nextHistory.pop()!;
      setHistory(nextHistory);
      setQueryId(prevId);
      window.location.hash = prevId;
    }
  };

  const handleHome = () => {
    if (items.length > 0) {
      const randomIndex = Math.floor(Math.random() * items.length);
      const randomId = items[randomIndex].id;
      if (queryId) {
        setHistory(prev => [...prev, queryId]);
      }
      setQueryId(randomId);
      window.location.hash = randomId;
    }
  };

  if (!mounted) {
    return <div className="w-screen h-screen bg-black" />;
  }

  // Developer/Empty state warning — also a drop target, so a first-time user
  // with no precomputed dataset can start populating the gallery immediately.
  if (items.length === 0) {
    return (
      <div
        className="w-screen h-screen bg-[#050505] flex flex-col items-center justify-center p-8 text-center"
        onDragOver={handleDragOver}
        onDragEnter={handleDragEnter}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <div className="max-w-md bg-neutral-900 border border-neutral-800 p-8 rounded-2xl shadow-2xl backdrop-blur-md">
          <h2 className="text-xl font-semibold mb-4 text-white">Welcome to River</h2>
          <p className="text-neutral-400 mb-6 text-sm leading-relaxed">
            Drag and drop images or videos here to try it out instantly — added for this session only.
          </p>
          <p className="text-neutral-400 mb-6 text-sm leading-relaxed">
            To build a permanent dataset instead, drop your image files (JPEG, PNG, WebP) into:
          </p>
          <code className="block bg-black text-emerald-400 p-3 rounded-lg text-xs font-mono mb-6 border border-neutral-800 text-left select-all">
            public/images/
          </code>
          <p className="text-neutral-400 mb-6 text-sm leading-relaxed">
            Then run the precomputation script to generate embeddings and thumbnails:
          </p>
          <code className="block bg-black text-emerald-400 p-3 rounded-lg text-xs font-mono mb-6 border border-neutral-800 text-left select-all">
            npm run precompute
          </code>
          <p className="text-neutral-500 text-xs">
            Once run, the embeddings will be saved to <span className="font-mono text-neutral-400">src/embeddings.json</span> and the app will refresh.
          </p>
        </div>
        {uploadStatus && <div className="status-info" style={{ position: "fixed", bottom: 24 }}>{uploadStatus}</div>}
        {isDraggingFile && (
          <div className="drop-zone-overlay">
            <div className="drop-zone-message">Drop images or videos to add them to this session</div>
          </div>
        )}
      </div>
    );
  }

  return (
    <div
      className="grid-container"
      onDragOver={handleDragOver}
      onDragEnter={handleDragEnter}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      {isDraggingFile && (
        <div className="drop-zone-overlay">
          <div className="drop-zone-message">Drop images or videos to add them to this session</div>
        </div>
      )}
      {/* Floating control overlay */}
      <div className="controls-panel">
        <button
          className="control-btn"
          onClick={handleBack}
          disabled={history.length === 0}
          title="Back to previous query"
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <line x1="19" y1="12" x2="5" y2="12"></line>
            <polyline points="12 19 5 12 12 5"></polyline>
          </svg>
          Back
        </button>
        <button
          className="control-btn"
          onClick={handleHome}
          title="Jump to random seed image"
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path>
            <polyline points="9 22 9 12 15 12 15 22"></polyline>
          </svg>
          Home
        </button>
        <button
          className="control-btn"
          onClick={() => setConvergeActive(true)}
          title="Converge: pick one aesthetic direction by pairwise choice"
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="4 7 9 12 4 17"></polyline>
            <polyline points="20 7 15 12 20 17"></polyline>
          </svg>
          Converge
        </button>
        <button
          className="control-btn"
          onClick={() => setNavigatorActive(true)}
          title="Explore embeddings as a 3D latent-space cloud"
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 2 21 7v10l-9 5-9-5V7z"></path>
            <path d="M12 2v20"></path>
            <path d="M21 7l-9 5-9-5"></path>
          </svg>
          Explore 3D
        </button>
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*,video/*"
          multiple
          className="visually-hidden"
          onChange={handleFileInputChange}
        />
        <button
          className="control-btn"
          onClick={() => fileInputRef.current?.click()}
          title="Add your own images or videos to this session"
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <line x1="12" y1="5" x2="12" y2="19"></line>
            <line x1="5" y1="12" x2="19" y2="12"></line>
          </svg>
          Add
        </button>
        <div className="divider"></div>

        {/* Content search: type words to find images by what they depict */}
        <div className="search-box">
          <svg className="search-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="11" cy="11" r="8"></circle>
            <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
          </svg>
          <input
            className="search-input"
            type="text"
            value={searchTerm}
            placeholder="Search by content…"
            onChange={(e) => setSearchTerm(e.target.value)}
            onKeyDown={handleSearchKeyDown}
            spellCheck={false}
          />
          {(searchTerm || textVec) && (
            <button className="search-clear" onClick={clearSearch} title="Clear search">
              ✕
            </button>
          )}
        </div>

        <div className="divider"></div>
        <button
          className="control-btn help-modal-trigger"
          onClick={() => setShowHelp(prev => !prev)}
          title="Show instructions"
        >
          ?
        </button>

        {/* Status display: upload progress, search progress, or item count */}
        <span className="status-info">
          {uploadStatus
            ? uploadStatus
            : searchStatus
              ? searchStatus
              : textVec
                ? `Results for “${searchTerm.trim()}”`
                : `${items.length} items`}
        </span>

        {/* Help popover */}
        <div ref={helpRef} className={`help-tooltip ${showHelp ? "visible" : ""}`}>
          <h3>River Exploration</h3>
          <p>This is a lateral exploration tool for your local image library, connected via SigLIP ViT-L/16 (384px) embeddings.</p>
          <ul>
            <li><span className="key-hint">Search</span> by content — type what you&apos;re looking for (e.g. &quot;buttons&quot;, &quot;a red flower&quot;) and press Enter to rank images by what they depict, not their filename.</li>
            <li><span className="key-hint">Left-Click</span> an image to refill the grid with its visually nearest neighbors.</li>
            <li><span className="key-hint">Right-Click</span> an image to open the high-res original in a new tab.</li>
            <li><span className="key-hint">Converge</span> mode — forced pairwise choices rank a diverse set by Bradley-Terry strength, then reveal which region you converged toward (←/→ or swipe; Esc to exit).</li>
            <li>Use the control buttons above to navigate your journey.</li>
          </ul>
          <p className="text-neutral-500 text-xs">The text model downloads once on your first search, then is cached for instant searches.</p>
          <p className="text-neutral-500 text-xs">Press anywhere outside to dismiss this panel.</p>
        </div>
      </div>

      {/* Main Justified Grid of absolute positioned elements */}
      {flatLaidOutItems.map(item => (
        <ImageCard
          key={item.id}
          item={item}
          onClick={() => handleImageClick(item.id)}
          onRightClick={(e) => handleImageRightClick(e, item.file)}
        />
      ))}

      {/* Converge overlay — mounts on demand, leaves the grid untouched */}
      {convergeActive && (
        <Converge items={items} onExit={() => setConvergeActive(false)} />
      )}

      {/* 3D latent-space navigator — mounts on demand, leaves the grid untouched */}
      {navigatorActive && (
        <Navigator3D items={items} onExit={() => setNavigatorActive(false)} />
      )}
    </div>
  );
}

interface ImageCardProps {
  item: LaidOutItem;
  onClick: () => void;
  onRightClick: (e: React.MouseEvent) => void;
}

const ImageCard: React.FC<ImageCardProps> = React.memo(({ item, onClick, onRightClick }) => {
  const [loaded, setLoaded] = useState(false);
  const [videoPlaying, setVideoPlaying] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const isVideo = item.kind === "video";

  useEffect(() => {
    let active = true;
    const img = new Image();
    img.src = item.thumb;

    if (img.complete) {
      setLoaded(true);
    } else {
      img.decode()
        .then(() => {
          if (active) {
            setLoaded(true);
          }
        })
        .catch(() => {
          img.onload = () => {
            if (active) {
              setLoaded(true);
            }
          };
        });
    }

    return () => {
      active = false;
    };
  }, [item.thumb]);

  // Hover-to-play: click and right-click are already bound to existing
  // interactions (hop-to-neighbor, open full-size), so hover is the only
  // affordance left for video preview — muted/looped, matching the common
  // silent-preview convention.
  const handleMouseEnter = () => {
    if (!isVideo || !videoRef.current) return;
    videoRef.current.currentTime = 0;
    videoRef.current.play().catch(() => {});
    setVideoPlaying(true);
  };
  const handleMouseLeave = () => {
    if (!isVideo || !videoRef.current) return;
    videoRef.current.pause();
    setVideoPlaying(false);
  };

  return (
    <div
      className="image-item"
      style={{
        left: `${item.left}px`,
        top: `${item.top}px`,
        width: `${item.width}px`,
        height: `${item.height}px`,
      }}
      onClick={onClick}
      onContextMenu={onRightClick}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
    >
      {isVideo ? (
        <>
          <video
            ref={videoRef}
            src={item.file}
            poster={item.thumb}
            className={`image-element ${loaded ? "loaded" : ""}`}
            muted
            loop
            playsInline
            preload="metadata"
          />
          {!videoPlaying && (
            <div className="video-play-icon">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                <polygon points="6 3 20 12 6 21 6 3"></polygon>
              </svg>
            </div>
          )}
        </>
      ) : (
        <img
          src={item.thumb}
          alt=""
          className={`image-element ${loaded ? "loaded" : ""}`}
          draggable={false}
        />
      )}
    </div>
  );
});

ImageCard.displayName = "ImageCard";
