"use client";

import React, { useState, useRef, useEffect, useMemo, useCallback } from "react";
import {
  type ConvergeItem,
  type ConvergeEngine,
  type AnalysisResult,
  createEngine,
  farthestPointSample,
  analyze,
  MAX_COMPARISONS,
  POOL_SIZE,
} from "./converge";

// The component needs render metadata on top of the raw vectors.
export interface ConvergeImage extends ConvergeItem {
  thumb: string;
  file: string;
  w: number;
  h: number;
}

interface ConvergeProps {
  items: ConvergeImage[];
  onExit: () => void;
}

type Phase = "compare" | "analyzing" | "done";

const QUESTION = "Which do you want your portfolio to look like?";

export default function Converge({ items, onExit }: ConvergeProps) {
  const byId = useMemo(() => new Map(items.map((it) => [it.id, it])), [items]);

  const engineRef = useRef<ConvergeEngine | null>(null);
  const [runId, setRunId] = useState(0);
  const [phase, setPhase] = useState<Phase>("compare");
  const [pair, setPair] = useState<[ConvergeImage, ConvergeImage] | null>(null);
  const [turn, setTurn] = useState(0);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [picked, setPicked] = useState<string | null>(null); // brief select highlight

  const resolvePair = useCallback(
    (engine: ConvergeEngine): [ConvergeImage, ConvergeImage] | null => {
      if (!engine.currentPair) return null;
      const a = byId.get(engine.currentPair[0].id);
      const b = byId.get(engine.currentPair[1].id);
      return a && b ? [a, b] : null;
    },
    [byId]
  );

  // (Re)initialize the engine on mount and on "Run again".
  useEffect(() => {
    const pool = farthestPointSample(items, POOL_SIZE);
    const engine = createEngine(pool, MAX_COMPARISONS);
    engineRef.current = engine;
    setResult(null);
    setPicked(null);
    setTurn(0);
    setPhase("compare");
    setPair(resolvePair(engine));
  }, [items, runId, resolvePair]);

  // Compute the analysis off the click handler so an "Analyzing…" frame paints.
  useEffect(() => {
    if (phase !== "analyzing") return;
    const engine = engineRef.current;
    if (!engine) return;
    const id = window.setTimeout(() => {
      setResult(analyze(engine, items));
      setPhase("done");
    }, 16);
    return () => window.clearTimeout(id);
  }, [phase, items]);

  const choose = useCallback(
    (winnerId: string) => {
      const engine = engineRef.current;
      if (!engine || !engine.currentPair || phase !== "compare") return;
      setPicked(winnerId);
      engine.choose(winnerId);
      setTurn(engine.comparisons.length);
      // Brief highlight, then advance.
      window.setTimeout(() => {
        setPicked(null);
        if (engine.isDone()) {
          setPair(null);
          setPhase("analyzing");
        } else {
          setPair(resolvePair(engine));
        }
      }, 140);
    },
    [phase, resolvePair]
  );

  // Keyboard: ←/→ choose, Esc exits.
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        onExit();
        return;
      }
      if (phase !== "compare" || !pair) return;
      if (e.key === "ArrowLeft") choose(pair[0].id);
      else if (e.key === "ArrowRight") choose(pair[1].id);
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [pair, phase, choose, onExit]);

  // Swipe toward the image you want.
  const touchStartX = useRef<number | null>(null);
  const onTouchStart = (e: React.TouchEvent) => {
    if (!e.touches || e.touches.length === 0) return;
    touchStartX.current = e.touches[0].clientX;
  };
  const onTouchEnd = (e: React.TouchEvent) => {
    if (touchStartX.current === null || !pair || phase !== "compare") return;
    if (!e.changedTouches || e.changedTouches.length === 0) {
      touchStartX.current = null;
      return;
    }
    const dx = e.changedTouches[0].clientX - touchStartX.current;
    touchStartX.current = null;
    if (Math.abs(dx) < 50) return;
    choose(dx < 0 ? pair[0].id : pair[1].id);
  };

  // -------------------------------------------------------------------------
  // Comparison view — two images + the question + a thin progress bar. No chrome.
  // -------------------------------------------------------------------------
  if (phase === "compare" || phase === "analyzing") {
    return (
      <div className="converge-overlay">
        <div className="converge-progress">
          <div
            className="converge-progress-fill"
            style={{ width: `${(turn / MAX_COMPARISONS) * 100}%` }}
          />
        </div>

        {pair && (
          <div
            className="converge-stage"
            onTouchStart={onTouchStart}
            onTouchEnd={onTouchEnd}
          >
            {[pair[0], pair[1]].map((img) => (
              <button
                key={img.id}
                className={`converge-choice ${picked === img.id ? "picked" : ""}`}
                onClick={() => choose(img.id)}
                aria-label="Choose this image"
              >
                <img src={img.thumb} alt="" draggable={false} />
              </button>
            ))}
          </div>
        )}

        <div className="converge-question">
          {phase === "analyzing" ? "Analyzing…" : QUESTION}
        </div>
      </div>
    );
  }

  // -------------------------------------------------------------------------
  // Analysis screen
  // -------------------------------------------------------------------------
  return (
    <div className="converge-overlay converge-results">
      <div className="converge-results-header">
        <h2>Converged</h2>
        <div className="converge-actions">
          <button className="converge-btn" onClick={() => setRunId((n) => n + 1)}>
            Run again
          </button>
          <button className="converge-btn" onClick={onExit}>
            Close
          </button>
        </div>
      </div>

      {result && (
        <div className="converge-results-body">
          {/* 1. Ranked list */}
          <section className="converge-section">
            <h3>Ranking</h3>
            <p className="converge-sub">
              Bradley-Terry strength over {MAX_COMPARISONS} forced choices.
            </p>
            <div className="converge-ranklist">
              {result.ranked.map((row) => {
                const img = byId.get(row.id);
                return (
                  <div key={row.id} className="converge-rankrow">
                    <span className="converge-rank-num">{row.rank}</span>
                    {img && (
                      <img className="converge-rank-thumb" src={img.thumb} alt="" />
                    )}
                    <div className="converge-rank-bar-wrap">
                      <div
                        className="converge-rank-bar"
                        style={{ width: `${Math.max(2, row.normScore)}%` }}
                      />
                    </div>
                    <span className="converge-rank-meta">
                      {row.normScore.toFixed(0)} · {row.wins}/{row.appearances}
                      {row.appearances > 0
                        ? ` · ${(row.winRate * 100).toFixed(0)}%`
                        : " · unseen"}
                    </span>
                  </div>
                );
              })}
            </div>
          </section>

          {/* 2. Cluster location */}
          {result.clusterLocation && (
            <section className="converge-section">
              <h3>Where you landed</h3>
              <p className="converge-sub">
                The centroid of your picks falls in cluster #
                {result.clusterLocation.clusterId} —{" "}
                <strong>{result.clusterLocation.label}</strong> (
                {result.clusterLocation.size} images, similarity{" "}
                {result.clusterLocation.similarity.toFixed(3)}).
              </p>
              {(() => {
                const rep = byId.get(result.clusterLocation.representativeId);
                return rep ? (
                  <img className="converge-cluster-rep" src={rep.thumb} alt="" />
                ) : null;
              })()}
            </section>
          )}

          {/* 3. In-between surface */}
          <section className="converge-section">
            <h3>In between</h3>
            <p className="converge-sub">
              Nearest to your direction across the full dataset — never directly
              compared.
            </p>
            <div className="converge-grid">
              {result.inBetween.map((id) => {
                const img = byId.get(id);
                return img ? (
                  <img
                    key={id}
                    className="converge-grid-thumb"
                    src={img.thumb}
                    alt=""
                    onClick={() => window.open(img.file, "_blank")}
                  />
                ) : null;
              })}
            </div>
          </section>

          {/* 4. Tension pair */}
          <section className="converge-section">
            <h3>Tension</h3>
            {result.tensionPair && result.tensionPair.upset > 0 ? (
              <>
                <p className="converge-sub">
                  Your biggest contradiction — you chose the left image, but the
                  final ranking rates the right one higher.
                </p>
                <div className="converge-tension">
                  {(() => {
                    const tp = result.tensionPair;
                    const w = byId.get(tp.winnerId);
                    const l = byId.get(tp.loserId);
                    return (
                      <>
                        <figure>
                          {w && <img src={w.thumb} alt="" />}
                          <figcaption>
                            you chose · {tp.winnerStrength.toFixed(2)}
                          </figcaption>
                        </figure>
                        <span className="converge-tension-vs">vs</span>
                        <figure>
                          {l && <img src={l.thumb} alt="" />}
                          <figcaption>
                            ranks higher · {tp.loserStrength.toFixed(2)}
                          </figcaption>
                        </figure>
                      </>
                    );
                  })()}
                </div>
              </>
            ) : (
              <p className="converge-sub">
                No contradictions — every choice agreed with the final ranking.
              </p>
            )}
          </section>
        </div>
      )}
    </div>
  );
}
