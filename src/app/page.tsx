"use client";

import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import dynamic from "next/dynamic";
import { embedImage, embedText, isTextModelLoaded } from "./embed";
import {
  fusedVector,
  loadSpace,
  placeUpload,
  rankByCluster,
  rankByText,
  rankByVector,
  type SpaceCluster,
  type SpaceItem,
} from "./space";

// The cloud pulls in three.js; load it only when opened.
const Cloud = dynamic(() => import("./Cloud"), { ssr: false });

type Query =
  | { kind: "image"; id: string }
  | { kind: "text"; text: string; vec: Float32Array }
  | { kind: "cluster"; index: number };

interface Tile {
  index: number;
  x: number;
  y: number;
  w: number;
  h: number;
  rank: number;
}

const GAP = 2;
const MIN_ASPECT = 0.4;
const MAX_ASPECT = 2.4;
const REVEAL_STEP_MS = 12; // stagger between tiles when a new result set appears
const REVEAL_MAX_RANK = 28; // tiles past this rank appear with the last group

// Justified rows in rank order. Row height tapers from 1.6× to 1× the base
// height, so the closest matches are also the largest.
function layoutGrid(order: number[], items: SpaceItem[], width: number): { tiles: Tile[]; height: number } {
  const base = Math.min(250, Math.max(140, width / 7));
  const target = (row: number) => base * (1 + 0.6 * Math.exp(-row / 1.5));
  // Extreme panoramas and strips are cropped (object-fit: cover) so one image
  // cannot squeeze a whole row into a sliver.
  const aspect = (i: number) => Math.min(MAX_ASPECT, Math.max(MIN_ASPECT, items[i].w / items[i].h || 1));
  const tiles: Tile[] = [];
  let y = 0;
  let row: number[] = [];
  let rowAspect = 0;
  let rowIndex = 0;
  const flush = (last: boolean) => {
    let h = (width - GAP * (row.length - 1)) / rowAspect;
    if (last) h = Math.min(h, target(rowIndex)); // an unfilled last row keeps its height
    let x = 0;
    for (const i of row) {
      const w = h * aspect(i);
      tiles.push({ index: i, x, y, w, h, rank: tiles.length });
      x += w + GAP;
    }
    y += h + GAP;
    rowIndex++;
    row = [];
    rowAspect = 0;
  };
  for (const i of order) {
    const a = aspect(i);
    if (row.length > 0) {
      const t = target(rowIndex);
      const without = (width - GAP * (row.length - 1)) / rowAspect;
      const withIt = (width - GAP * row.length) / (rowAspect + a);
      if (withIt < t && Math.abs(without - t) < Math.abs(withIt - t)) flush(false);
    }
    row.push(i);
    rowAspect += a;
  }
  if (row.length > 0) flush(true);
  return { tiles, height: Math.max(0, y - GAP) };
}

const VIDEO_POSTER_SEEK_FRACTION = 0.1; // frame 0 is often black

// One representative frame of a video, without a server round-trip.
async function extractVideoPosterFrame(file: File): Promise<{ videoUrl: string; poster: Blob; w: number; h: number }> {
  const videoUrl = URL.createObjectURL(file);
  const video = document.createElement("video");
  video.src = videoUrl;
  video.muted = true;
  video.playsInline = true;
  await new Promise<void>((resolve, reject) => {
    video.onloadedmetadata = () => resolve();
    video.onerror = () => reject(new Error("Failed to load video metadata"));
  });
  await new Promise<void>((resolve, reject) => {
    video.onseeked = () => resolve();
    video.onerror = () => reject(new Error("Failed to seek video"));
    video.currentTime = (video.duration || 0) * VIDEO_POSTER_SEEK_FRACTION;
  });
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext("2d")?.drawImage(video, 0, 0);
  const poster = await new Promise<Blob>((resolve, reject) =>
    canvas.toBlob((b) => (b ? resolve(b) : reject(new Error("Canvas toBlob failed"))), "image/webp", 0.9)
  );
  return { videoUrl, poster, w: video.videoWidth, h: video.videoHeight };
}

let uploadSeq = 0;

export default function Home() {
  const [space] = useState(loadSpace);
  const clusters: SpaceCluster[] = space.clusters;
  const [items, setItems] = useState<SpaceItem[]>(space.items);
  const [mounted, setMounted] = useState(false);
  const [query, setQuery] = useState<Query | null>(null);
  const [history, setHistory] = useState<Query[]>([]);
  const [revealKey, setRevealKey] = useState(0);
  const [width, setWidth] = useState(1200);
  const [cloudOpen, setCloudOpen] = useState(false);

  const [searchTerm, setSearchTerm] = useState("");
  const [status, setStatus] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);

  const mainRef = useRef<HTMLElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const searchSeq = useRef(0);
  const dragDepth = useRef(0);

  const indexById = useMemo(() => new Map(items.map((it, i) => [it.id, i])), [items]);
  const queryRef = useRef<Query | null>(null);
  queryRef.current = query;
  const hashRef = useRef(""); // the hash this page set last, so its own change is ignored

  // Every query change goes through here: it records history, puts image
  // queries in the URL hash (browser back/forward works), and restarts the
  // reveal from the top of the page.
  const go = useCallback((next: Query, record = true) => {
    const prev = queryRef.current;
    if (record && prev) setHistory((h) => [...h, prev]);
    setQuery(next);
    setSearchTerm(next.kind === "text" ? next.text : "");
    if (next.kind === "image" && window.location.hash.slice(1) !== next.id) {
      hashRef.current = next.id;
      window.location.hash = next.id;
    }
    setRevealKey((k) => k + 1);
    window.scrollTo({ top: 0 });
  }, []);

  const randomImage = useCallback((): Query => {
    return { kind: "image", id: items[Math.floor(Math.random() * items.length)].id };
  }, [items]);

  useEffect(() => {
    setMounted(true);
    const hash = decodeURIComponent(window.location.hash.slice(1));
    const start = space.items.find((it) => it.id === hash) ?? space.items[Math.floor(Math.random() * space.items.length)];
    setQuery({ kind: "image", id: start.id });
  }, [space]);

  useEffect(() => {
    const onHash = () => {
      const id = decodeURIComponent(window.location.hash.slice(1));
      if (id === hashRef.current) return;
      hashRef.current = id;
      if (indexById.has(id)) go({ kind: "image", id }, false);
    };
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, [go, indexById]);

  useEffect(() => {
    const el = mainRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setWidth(el.clientWidth));
    ro.observe(el);
    setWidth(el.clientWidth);
    return () => ro.disconnect();
  }, [mounted]);

  const order = useMemo(() => {
    if (!query) return items.map((_, i) => i);
    if (query.kind === "text") return rankByText(query.vec, items);
    if (query.kind === "cluster") return rankByCluster(query.index, items);
    const q = indexById.get(query.id);
    if (q === undefined) return items.map((_, i) => i);
    return [q, ...rankByVector(items[q].fused, items).filter((i) => i !== q)];
  }, [query, items, indexById]);

  const grid = useMemo(() => layoutGrid(order, items, width), [order, items, width]);

  const queryItem = query?.kind === "image" ? items[indexById.get(query.id) ?? -1] : undefined;

  const openImage = useCallback(
    (id: string) => {
      searchSeq.current++; // a search still loading must not replace this
      go({ kind: "image", id });
    },
    [go]
  );

  const back = () => {
    const prev = history[history.length - 1];
    if (!prev) return;
    setHistory((h) => h.slice(0, -1));
    go(prev, false);
  };

  // ---- text search ------------------------------------------------------------

  const runSearch = async (text: string) => {
    const seq = ++searchSeq.current;
    if (!text.trim()) return;
    try {
      setStatus(isTextModelLoaded() ? "Searching…" : "Loading text model (once, about 280 MB)…");
      const vec = await embedText(text);
      if (searchSeq.current !== seq) return;
      setStatus(null);
      go({ kind: "text", text: text.trim(), vec });
    } catch (err) {
      console.error("Text search failed:", err);
      if (searchSeq.current === seq) setStatus("Search failed");
    }
  };

  const clearSearch = () => {
    searchSeq.current++;
    setSearchTerm("");
    setStatus(null);
    if (query?.kind === "text") back();
  };

  // ---- uploads ----------------------------------------------------------------

  const ingest = async (files: File[]) => {
    const media = files.filter((f) => f.type.startsWith("image/") || f.type.startsWith("video/"));
    if (media.length === 0) return;
    const added: SpaceItem[] = [];
    let done = 0;
    setStatus(`Embedding 0 / ${media.length}…`);
    try {
      for (const file of media) {
        const isVideo = file.type.startsWith("video/");
        let blob: Blob = file;
        let videoUrl: string | undefined;
        let w = 0;
        let h = 0;
        if (isVideo) {
          const frame = await extractVideoPosterFrame(file);
          blob = frame.poster;
          videoUrl = frame.videoUrl;
          w = frame.w;
          h = frame.h;
        } else {
          const bitmap = await createImageBitmap(file);
          w = bitmap.width;
          h = bitmap.height;
          bitmap.close();
        }
        const { siglip, dino } = await embedImage(blob);
        const fused = fusedVector(siglip, dino);
        const base = [...items, ...added];
        const place = placeUpload(fused, base);
        added.push({
          id: `upload-${Date.now()}-${++uploadSeq}`,
          thumb: URL.createObjectURL(blob),
          file: videoUrl,
          w,
          h,
          kind: isVideo ? "video" : "image",
          siglip,
          fused,
          tags: [],
          ...place,
        });
        done++;
        setStatus(`Embedding ${done} / ${media.length}…`);
      }
      setItems((prev) => [...prev, ...added]);
      setStatus(null);
      if (added.length > 0) openImage(added[0].id);
    } catch (err) {
      console.error("Upload failed:", err);
      setStatus("Upload failed");
    }
  };

  const onDragEnter = (e: React.DragEvent) => {
    e.preventDefault();
    if (!e.dataTransfer.types.includes("Files")) return;
    dragDepth.current++;
    setDragging(true);
  };
  const onDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    dragDepth.current = Math.max(0, dragDepth.current - 1);
    if (dragDepth.current === 0) setDragging(false);
  };
  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    dragDepth.current = 0;
    setDragging(false);
    ingest(Array.from(e.dataTransfer.files));
  };

  if (!mounted) return <div className="app" />;

  const cluster = queryItem ? clusters[queryItem.cluster] : query?.kind === "cluster" ? clusters[query.index] : undefined;
  const parent = cluster && cluster.parent !== null ? clusters[cluster.parent] : undefined;

  return (
    <div
      className="app"
      onDragOver={(e) => e.preventDefault()}
      onDragEnter={onDragEnter}
      onDragLeave={onDragLeave}
      onDrop={onDrop}
    >
      <header className="bar">
        <div className="bar-group">
          <button className="btn" onClick={back} disabled={history.length === 0} title="Previous result" aria-label="Back">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="15 18 9 12 15 6" />
            </svg>
            <span className="btn-label">Back</span>
          </button>
          <button className="btn" onClick={() => go(randomImage())} title="Start from a random image" aria-label="Shuffle">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="16 3 21 3 21 8" />
              <line x1="4" y1="20" x2="21" y2="3" />
              <polyline points="21 16 21 21 16 21" />
              <line x1="15" y1="15" x2="21" y2="21" />
              <line x1="4" y1="4" x2="9" y2="9" />
            </svg>
            <span className="btn-label">Shuffle</span>
          </button>
          <button className="btn btn-primary" onClick={() => setCloudOpen(true)} title="See the board as a cloud of aesthetics">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="7" cy="12" r="2" />
              <circle cx="16" cy="7" r="2" />
              <circle cx="17" cy="16" r="2" />
            </svg>
            Cloud
          </button>
        </div>
        <form
          className="search"
          onSubmit={(e) => {
            e.preventDefault();
            runSearch(searchTerm);
          }}
        >
          <svg className="search-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
            <circle cx="11" cy="11" r="7" />
            <line x1="20" y1="20" x2="16.5" y2="16.5" />
          </svg>
          <input
            className="search-input"
            type="text"
            value={searchTerm}
            placeholder="Search by content or aesthetic"
            onChange={(e) => setSearchTerm(e.target.value)}
            spellCheck={false}
            aria-label="Search images"
          />
          {searchTerm && (
            <button type="button" className="search-clear" onClick={clearSearch} aria-label="Clear search">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                <line x1="6" y1="6" x2="18" y2="18" />
                <line x1="18" y1="6" x2="6" y2="18" />
              </svg>
            </button>
          )}
        </form>
        <div className="bar-group bar-end">
          <span className="status">{status ?? `${items.length} images`}</span>
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*,video/*"
            multiple
            hidden
            onChange={(e) => {
              ingest(Array.from(e.target.files ?? []));
              e.target.value = "";
            }}
          />
          <button className="btn" onClick={() => fileInputRef.current?.click()} title="Add images or videos to this session" aria-label="Add">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
              <line x1="12" y1="5" x2="12" y2="19" />
              <line x1="5" y1="12" x2="19" y2="12" />
            </svg>
            <span className="btn-label">Add</span>
          </button>
        </div>
      </header>

      <main ref={mainRef} className="main">
        <div className="context">
          {query?.kind === "text" ? (
            <>
              <span className="meta">Search</span>
              <span className="context-title">{query.text}</span>
            </>
          ) : cluster ? (
            <>
              {parent && <span className="meta">{parent.label}</span>}
              {query?.kind === "image" ? (
                <button className="context-title context-link" onClick={() => go({ kind: "cluster", index: queryItem!.cluster })}>
                  {cluster.label}
                </button>
              ) : (
                <span className="context-title">{cluster.label}</span>
              )}
              {query?.kind === "image" && queryItem!.tags.length > 0 && (
                <span className="context-tags">{queryItem!.tags.join(" · ")}</span>
              )}
              {query?.kind === "cluster" && <span className="meta">{cluster.size} images</span>}
            </>
          ) : null}
        </div>
        <div className="grid" style={{ height: grid.height }} key={revealKey}>
          {grid.tiles.map((t) => {
            const it = items[t.index];
            return (
              <TileView
                key={it.id}
                item={it}
                tile={t}
                isQuery={query?.kind === "image" && it.id === query.id}
                label={it.tags[0] ?? clusters[it.cluster]?.label ?? ""}
                onOpen={openImage}
              />
            );
          })}
        </div>
      </main>

      {cloudOpen && (
        <Cloud
          items={items}
          clusters={clusters}
          focusId={queryItem?.id ?? null}
          onPick={(id) => {
            setCloudOpen(false);
            openImage(id);
          }}
          onClose={() => setCloudOpen(false)}
        />
      )}

      {dragging && (
        <div className="drop-overlay">
          <p>Drop images or videos to add them to this session</p>
        </div>
      )}
    </div>
  );
}

interface TileProps {
  item: SpaceItem;
  tile: Tile;
  isQuery: boolean;
  label: string;
  onOpen: (id: string) => void;
}

const TileView = React.memo(function TileView({ item, tile, isQuery, label, onOpen }: TileProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const isVideo = item.kind === "video";
  return (
    <button
      className={`tile${isQuery ? " tile-query" : ""}`}
      style={
        {
          transform: `translate(${tile.x}px, ${tile.y}px)`,
          width: tile.w,
          height: tile.h,
          "--reveal-delay": `${Math.min(tile.rank, REVEAL_MAX_RANK) * REVEAL_STEP_MS}ms`,
        } as React.CSSProperties
      }
      onClick={() => onOpen(item.id)}
      onMouseEnter={() => {
        if (isVideo && videoRef.current) videoRef.current.play().catch(() => {});
      }}
      onMouseLeave={() => {
        if (isVideo && videoRef.current) videoRef.current.pause();
      }}
      aria-label={item.tags.length > 0 ? item.tags.join(", ") : label}
    >
      {isVideo ? (
        <video ref={videoRef} src={item.file} poster={item.thumb} muted loop playsInline preload="metadata" />
      ) : (
        <img src={item.thumb} alt="" draggable={false} decoding="async" loading={tile.rank < 24 ? "eager" : "lazy"} />
      )}
      {label && <span className="tile-label">{label}</span>}
    </button>
  );
});
