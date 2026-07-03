"use client";

// Steer pad — a floating 2D controller that turns puck position into a live
// query vector for the existing similarity pipeline (math in steer.ts).
//
// Rendering discipline: React only renders the chrome (labels, status). The
// drag path — puck transform, readout text, query/score/relevance compute —
// runs through refs and requestAnimationFrame, so a drag causes ZERO React
// commits. Pole edits re-derive axes on blur/Enter via the shared SigLIP text
// tower passed in as embedText.

import React, { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import {
  axisStrength,
  buildSteerQuery,
  deriveAxis,
  orthogonalize,
  scoreAll,
  scoresToRelevance,
  type SteerContext,
} from "./steer";

// Default word poles: X = left→right, Y = bottom→top. All four are editable.
const DEFAULT_POLES = {
  left: "minimal",
  right: "ornate",
  bottom: "dark",
  top: "vibrant",
};

// Keep the puck's center this many px inside the surface edge.
const PUCK_INSET = 10;

type Poles = typeof DEFAULT_POLES;
type PoleSide = keyof Poles;
type Status = "loading" | "ready" | "error" | "degenerate-x" | "degenerate-y";

const STATUS_LABEL: Record<Status, string> = {
  loading: "deriving axes…",
  ready: "",
  error: "axes failed — edit a pole",
  "degenerate-x": "x poles too similar",
  "degenerate-y": "y poles too similar",
};

interface SteerPadProps {
  ctx: SteerContext | null; // null → pad renders disabled (no items yet)
  embedText: (text: string) => Promise<Float32Array>;
  onSteer: (relevance: Float32Array, topIndex: number, query: Float32Array) => void;
  onSteerEnd: () => void;
}

interface Axes {
  ax: Float32Array | null;
  ay: Float32Array | null;
  sx: number;
  sy: number;
}

interface Scratch {
  query: Float32Array;
  scores: Float32Array;
  relevance: Float32Array;
}

const clamp1 = (v: number) => Math.max(-1, Math.min(1, v));
const fmt = (v: number) => `${v < 0 ? "−" : "+"}${Math.abs(v).toFixed(2)}`;

export default function SteerPad({ ctx, embedText, onSteer, onSteerEnd }: SteerPadProps) {
  const [poles, setPoles] = useState<Poles>(DEFAULT_POLES);
  const [status, setStatus] = useState<Status>("loading");

  const surfaceRef = useRef<HTMLDivElement>(null);
  const puckRef = useRef<HTMLDivElement>(null);
  const readoutRef = useRef<HTMLSpanElement>(null);

  const xyRef = useRef({ x: 0, y: 0 });
  const rafRef = useRef(0);
  const draggingRef = useRef(false);
  const rectRef = useRef<{ left: number; top: number; w: number; h: number } | null>(null);
  const axesRef = useRef<Axes>({ ax: null, ay: null, sx: 0, sy: 0 });
  const scratchRef = useRef<Scratch | null>(null);
  // Word → embedding, so editing one pole embeds only the new word.
  const wordVecCacheRef = useRef(new Map<string, Float32Array>());
  // Pole words of the last successful derivation (Escape reverts to these;
  // blur with unchanged words is a no-op).
  const lastDerivedRef = useRef<Poles | null>(null);
  // Monotonic token — a stale async derivation resolving late becomes a no-op
  // (same pattern as page.tsx's searchSeqRef).
  const deriveSeqRef = useRef(0);

  // Latest-value refs so the once-created rAF/pointer closures never go stale
  // (mirrors Navigator3D's focusedIdRef pattern).
  const ctxRef = useRef(ctx);
  ctxRef.current = ctx;
  const embedTextRef = useRef(embedText);
  embedTextRef.current = embedText;
  const onSteerRef = useRef(onSteer);
  onSteerRef.current = onSteer;
  const onSteerEndRef = useRef(onSteerEnd);
  onSteerEndRef.current = onSteerEnd;

  // Resize scratch buffers to the current context and refresh per-axis
  // strengths (std of projections depends on the images, not the words).
  const syncToContext = useCallback(() => {
    const c = ctxRef.current;
    if (!c || c.n === 0) {
      scratchRef.current = null;
      return;
    }
    const s = scratchRef.current;
    if (!s || s.query.length !== c.dim || s.scores.length !== c.n) {
      scratchRef.current = {
        query: new Float32Array(c.dim),
        scores: new Float32Array(c.n),
        relevance: new Float32Array(c.n),
      };
    }
    const a = axesRef.current;
    a.sx = a.ax ? axisStrength(c, a.ax) : 0;
    a.sy = a.ay ? axisStrength(c, a.ay) : 0;
  }, []);

  // One compute pass at the current puck position: move the puck, update the
  // readout, and emit relevance. Zero-alloc — everything writes into scratch.
  const frame = useCallback(() => {
    rafRef.current = 0;
    const { x, y } = xyRef.current;
    const rect = rectRef.current;
    if (puckRef.current && rect) {
      const px = x * (rect.w / 2 - PUCK_INSET);
      const py = -y * (rect.h / 2 - PUCK_INSET);
      puckRef.current.style.transform = `translate(${px.toFixed(1)}px, ${py.toFixed(1)}px)`;
    }
    if (readoutRef.current) readoutRef.current.textContent = `${fmt(x)} · ${fmt(y)}`;

    const c = ctxRef.current;
    const s = scratchRef.current;
    const { ax, ay, sx, sy } = axesRef.current;
    if (!c || !s || (!ax && !ay)) return; // puck still moves before axes exist
    buildSteerQuery(c, ax, ay, x, y, sx, sy, s.query);
    scoreAll(s.query, c, s.scores);
    const top = scoresToRelevance(s.scores, c.n, s.relevance);
    onSteerRef.current(s.relevance, top, s.query);
  }, []);

  const scheduleFrame = useCallback(() => {
    if (rafRef.current === 0) rafRef.current = requestAnimationFrame(frame);
  }, [frame]);

  // Embed the four pole words (cached per word) and rebuild the axes.
  const deriveAxes = useCallback(async (p: Poles) => {
    const seq = ++deriveSeqRef.current;
    setStatus("loading");
    try {
      const words: Record<PoleSide, string> = {
        left: p.left.trim(),
        right: p.right.trim(),
        bottom: p.bottom.trim(),
        top: p.top.trim(),
      };
      // Embed all missing words in parallel — the model load is deduped
      // upstream (ensureTextModel memoizes its promise) and ONNX serializes
      // the runs internally, so this only removes await-chain latency.
      const missing = [...new Set(Object.values(words).filter(w => w && !wordVecCacheRef.current.has(w)))];
      await Promise.all(
        missing.map(async (w) => {
          const v = await embedTextRef.current(w);
          wordVecCacheRef.current.set(w, v);
        })
      );
      if (deriveSeqRef.current !== seq) return;
      const vecs = {} as Record<PoleSide, Float32Array | null>;
      for (const side of ["left", "right", "bottom", "top"] as PoleSide[]) {
        // Empty pole → that axis disabled, 1-axis steering.
        vecs[side] = words[side] ? wordVecCacheRef.current.get(words[side])! : null;
      }

      let degenerate: "x" | "y" | null = null;
      let ax: Float32Array | null = null;
      let ay: Float32Array | null = null;
      if (vecs.left && vecs.right) {
        ax = deriveAxis(vecs.left, vecs.right);
        if (!ax) degenerate = "x";
      }
      if (vecs.bottom && vecs.top) {
        const rawY = deriveAxis(vecs.bottom, vecs.top);
        ay = rawY && ax ? orthogonalize(rawY, ax) : rawY;
        if (!ay) degenerate = degenerate ?? "y";
      }
      // Degenerate poles keep the previous (still meaningful) axis.
      const prev = axesRef.current;
      if (!ax && degenerate === "x") ax = prev.ax;
      if (!ay && degenerate === "y") ay = prev.ay;
      axesRef.current = { ax, ay, sx: 0, sy: 0 };
      syncToContext();
      lastDerivedRef.current = p;
      setStatus(degenerate === "x" ? "degenerate-x" : degenerate === "y" ? "degenerate-y" : "ready");
      frame(); // refresh visuals at the current puck position without a drag
    } catch (err) {
      console.error("Steer axis derivation failed:", err);
      if (deriveSeqRef.current === seq) setStatus("error");
    }
  }, [frame, syncToContext]);

  // Mount: measure the surface, center the readout, derive the default axes.
  useLayoutEffect(() => {
    const surface = surfaceRef.current;
    if (surface) {
      const r = surface.getBoundingClientRect();
      rectRef.current = { left: r.left, top: r.top, w: r.width, h: r.height };
    }
    frame();
    deriveAxes(DEFAULT_POLES);
    const raf = rafRef;
    const seq = deriveSeqRef;
    return () => {
      if (raf.current) cancelAnimationFrame(raf.current);
      seq.current++; // invalidate any in-flight derivation
    };
    // Mount-only by design; state is reached through refs.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Items changed (upload) or context arrived: re-pack scratch + strengths and
  // repaint. The axes themselves are item-independent.
  useEffect(() => {
    syncToContext();
    frame();
  }, [ctx, syncToContext, frame]);

  // ---- drag ----------------------------------------------------------------

  const updateFromPointer = (e: React.PointerEvent) => {
    const rect = rectRef.current;
    if (!rect) return;
    xyRef.current.x = clamp1(((e.clientX - rect.left) / rect.w) * 2 - 1);
    xyRef.current.y = clamp1(-(((e.clientY - rect.top) / rect.h) * 2 - 1));
  };

  const handlePointerDown = (e: React.PointerEvent) => {
    if (e.target instanceof HTMLInputElement) return; // pole edits, not drags
    const surface = surfaceRef.current;
    if (!surface) return;
    // Re-measure on grab — the one layout read of the whole gesture.
    const r = surface.getBoundingClientRect();
    rectRef.current = { left: r.left, top: r.top, w: r.width, h: r.height };
    draggingRef.current = true;
    surface.setPointerCapture(e.pointerId);
    updateFromPointer(e);
    scheduleFrame();
  };

  const handlePointerMove = (e: React.PointerEvent) => {
    if (!draggingRef.current) return;
    updateFromPointer(e);
    scheduleFrame(); // coalesces bursts of moves into one compute per frame
  };

  const endDrag = (e: React.PointerEvent) => {
    if (!draggingRef.current) return;
    draggingRef.current = false;
    surfaceRef.current?.releasePointerCapture(e.pointerId);
    if (rafRef.current) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = 0;
    }
    frame(); // final synchronous pass so the release state is exact
    onSteerEndRef.current();
  };

  // ---- pole label editing ----------------------------------------------------

  const commitPole = () => {
    const last = lastDerivedRef.current;
    const changed =
      !last ||
      (["left", "right", "bottom", "top"] as PoleSide[]).some(
        (side) => poles[side].trim() !== last[side].trim()
      );
    if (changed) deriveAxes(poles);
  };

  const handlePoleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      e.preventDefault();
      e.currentTarget.blur(); // blur commits
    } else if (e.key === "Escape") {
      setPoles(lastDerivedRef.current ?? DEFAULT_POLES); // revert, no re-derive
      e.currentTarget.blur();
    }
  };

  const poleInput = (side: PoleSide) => (
    <input
      className={`steer-pole ${side}`}
      type="text"
      value={poles[side]}
      size={Math.max(3, poles[side].length)}
      spellCheck={false}
      onChange={(e) => setPoles((prev) => ({ ...prev, [side]: e.target.value }))}
      onBlur={commitPole}
      onKeyDown={handlePoleKeyDown}
      aria-label={`${side} pole word`}
    />
  );

  return (
    <div className={`steer-pad${ctx ? "" : " disabled"}`}>
      <div
        ref={surfaceRef}
        className="steer-pad-surface"
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={endDrag}
        onPointerCancel={endDrag}
      >
        {poleInput("top")}
        {poleInput("bottom")}
        {poleInput("left")}
        {poleInput("right")}
        <div ref={puckRef} className="steer-puck" />
      </div>
      <div className="steer-readout">
        <span className={`steer-status${status === "error" ? " error" : ""}`}>
          {STATUS_LABEL[status]}
        </span>
        <span ref={readoutRef}>+0.00 · +0.00</span>
      </div>
    </div>
  );
}
