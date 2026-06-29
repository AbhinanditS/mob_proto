"use client";

import React, { useState, useEffect, useRef, useMemo } from "react";
import type { PreTrainedTokenizer, PreTrainedModel } from "@huggingface/transformers";
import rawEmbeddings from "@/embeddings.json";
import Converge from "./Converge";
import Navigator3D from "./Navigator3D";

interface ImageRecord {
  id: string;
  file: string;
  thumb: string;
  w: number;
  h: number;
  embedding: number[];
}

interface LaidOutItem extends ImageRecord {
  left: number;
  top: number;
  width: number;
  height: number;
}

// SigLIP checkpoint — must match scripts/precompute.ts so image and text
// embeddings live in the same vector space.
const MODEL = "Xenova/siglip-base-patch16-384";

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

  const helpRef = useRef<HTMLDivElement>(null);
  // Cache the lazily-loaded tokenizer + text model so only the first search
  // pays the load/download cost.
  const textModelRef = useRef<{ tokenizer: PreTrainedTokenizer; model: PreTrainedModel } | null>(null);
  // Monotonic token to invalidate in-flight searches when the user clears the
  // box, submits a new query, or clicks away before the previous one resolves.
  const searchSeqRef = useRef(0);

  // Parse items and cache their Float32Array embeddings
  const items = useMemo(() => {
    return (rawEmbeddings as ImageRecord[]).map(item => ({
      ...item,
      vec: new Float32Array(item.embedding)
    }));
  }, []);

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

  // Developer/Empty state warning
  if (items.length === 0) {
    return (
      <div className="w-screen h-screen bg-[#050505] flex flex-col items-center justify-center p-8 text-center">
        <div className="max-w-md bg-neutral-900 border border-neutral-800 p-8 rounded-2xl shadow-2xl backdrop-blur-md">
          <h2 className="text-xl font-semibold mb-4 text-white">Welcome to River</h2>
          <p className="text-neutral-400 mb-6 text-sm leading-relaxed">
            The dataset is currently empty. Drop your image files (JPEG, PNG, WebP) into:
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
      </div>
    );
  }

  return (
    <div className="grid-container">
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

        {/* Status display: search progress, or item count */}
        <span className="status-info">
          {searchStatus
            ? searchStatus
            : textVec
              ? `Results for “${searchTerm.trim()}”`
              : `${items.length} items`}
        </span>

        {/* Help popover */}
        <div ref={helpRef} className={`help-tooltip ${showHelp ? "visible" : ""}`}>
          <h3>River Exploration</h3>
          <p>This is a lateral exploration tool for your local image library, connected via SigLIP ViT-B/16 (384px) embeddings.</p>
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
    >
      <img
        src={item.thumb}
        alt=""
        className={`image-element ${loaded ? "loaded" : ""}`}
        draggable={false}
      />
    </div>
  );
});

ImageCard.displayName = "ImageCard";
