"use client";

// Full-screen shell around CloudRenderer (src/app/cloud.ts). Loaded on demand
// with next/dynamic, so three.js stays out of the grid's bundle.

import React, { useEffect, useRef } from "react";
import { CloudRenderer } from "./cloud";
import type { SpaceCluster, SpaceItem } from "./space";

interface CloudProps {
  items: SpaceItem[];
  clusters: SpaceCluster[];
  focusId: string | null; // the grid's current query image, ringed in the cloud
  onPick: (id: string) => void;
  onClose: () => void;
}

export default function Cloud({ items, clusters, focusId, onPick, onClose }: CloudProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rendererRef = useRef<CloudRenderer | null>(null);
  const onPickRef = useRef(onPick);
  onPickRef.current = onPick;
  const onCloseRef = useRef(onClose);
  onCloseRef.current = onClose;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const renderer = new CloudRenderer(canvas, { onPick: (id) => onPickRef.current(id) });
    rendererRef.current = renderer;
    const onResize = () => renderer.resize();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onCloseRef.current();
    };
    window.addEventListener("resize", onResize);
    window.addEventListener("keydown", onKey);
    // Canvas text needs the web fonts; redraw once they are ready.
    document.fonts?.ready.then(() => renderer.readTheme());
    return () => {
      window.removeEventListener("resize", onResize);
      window.removeEventListener("keydown", onKey);
      renderer.dispose();
      rendererRef.current = null;
    };
  }, []);

  useEffect(() => {
    rendererRef.current?.setData(items, clusters, focusId);
  }, [items, clusters, focusId]);

  return (
    <div className="cloud" role="dialog" aria-label="Cloud of aesthetics">
      <canvas ref={canvasRef} className="cloud-canvas" />
      <button className="cloud-close" onClick={onClose} title="Close (Esc)" aria-label="Close">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
          <line x1="6" y1="6" x2="18" y2="18" />
          <line x1="18" y1="6" x2="6" y2="18" />
        </svg>
      </button>
      <p className="cloud-hint">Hover to see images · Drag to rotate · Scroll to zoom · Click an image for similar</p>
    </div>
  );
}
