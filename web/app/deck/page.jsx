"use client";

import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import slides from "./slides";
import "./deck.css";

/**
 * The deck shell: a fixed 1280x720 stage scaled to the window, arrow-key
 * navigation, and an overview grid.
 *
 * Scaling rather than reflowing is deliberate — see deck.css. It also makes
 * the overview grid free: a thumbnail is the same slide at a smaller scale,
 * not a separate rendering.
 */

const STAGE_W = 1280;
const STAGE_H = 720;

function slugFor(i) {
  return slides[i]?.id ?? String(i);
}

function indexFromHash() {
  if (typeof window === "undefined") return 0;
  const h = window.location.hash.replace(/^#/, "");
  if (!h) return 0;
  const byId = slides.findIndex((s) => s.id === h);
  if (byId >= 0) return byId;
  const n = Number.parseInt(h, 10);
  if (Number.isFinite(n) && n >= 1 && n <= slides.length) return n - 1;
  return 0;
}

export default function Deck() {
  const [i, setI] = useState(0);
  const [grid, setGrid] = useState(false);
  const [scale, setScale] = useState(1);
  const [thumbScale, setThumbScale] = useState(0.18);
  const gridRef = useRef(null);

  // The hash is the deep link: /deck#p-winner opens on that slide, which is
  // how you recover mid-talk after a reload.
  useEffect(() => {
    setI(indexFromHash());
    const onHash = () => setI(indexFromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const go = useCallback((next) => {
    const clamped = Math.max(0, Math.min(slides.length - 1, next));
    setI(clamped);
    // replaceState rather than setting location.hash: pushing a history entry
    // per slide makes the browser Back button useless during a talk.
    if (typeof window !== "undefined") {
      window.history.replaceState(null, "", `#${slugFor(clamped)}`);
    }
  }, []);

  // Fit the stage to the window. useLayoutEffect so the first paint is already
  // at the right size rather than snapping.
  useLayoutEffect(() => {
    const fit = () => {
      const pad = 0;
      const s = Math.min(
        (window.innerWidth - pad) / STAGE_W,
        (window.innerHeight - pad) / STAGE_H,
      );
      setScale(s);
    };
    fit();
    window.addEventListener("resize", fit);
    return () => window.removeEventListener("resize", fit);
  }, []);

  // Thumbnails are the real slides scaled down, so they need the measured
  // column width — a hardcoded scale leaves a gap on the right of every card.
  useEffect(() => {
    if (!grid) return;
    const el = gridRef.current;
    if (!el) return;
    const measure = () => {
      const first = el.firstElementChild;
      if (first) setThumbScale(first.clientWidth / STAGE_W);
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, [grid]);

  useEffect(() => {
    const onKey = (e) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      switch (e.key) {
        case "ArrowRight":
        case "ArrowDown":
        case "PageDown":
        case " ":
          e.preventDefault();
          if (grid) setGrid(false);
          else go(i + 1);
          break;
        case "ArrowLeft":
        case "ArrowUp":
        case "PageUp":
          e.preventDefault();
          if (grid) setGrid(false);
          else go(i - 1);
          break;
        case "Home":
          e.preventDefault();
          go(0);
          break;
        case "End":
          e.preventDefault();
          go(slides.length - 1);
          break;
        case "g":
        case "o":
        case "Escape":
          e.preventDefault();
          setGrid((g) => (e.key === "Escape" ? false : !g));
          break;
        case "f":
          e.preventDefault();
          if (document.fullscreenElement) document.exitFullscreen();
          else document.documentElement.requestFullscreen?.();
          break;
        default:
          break;
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [i, grid, go]);

  const current = slides[i];
  const showFooter = Boolean(current?.section);

  return (
    <div className="deck-root">
      <div className="deck-hud">
        <button type="button" onClick={() => setGrid((g) => !g)}>
          {grid ? "Close" : "Overview"}
        </button>
        <button
          type="button"
          onClick={() => {
            if (document.fullscreenElement) document.exitFullscreen();
            else document.documentElement.requestFullscreen?.();
          }}
        >
          Full screen
        </button>
        <button type="button" onClick={() => { window.location.href = "/"; }}>
          Live demo
        </button>
      </div>

      {grid ? (
        <div className="deck-grid" ref={gridRef}>
          {slides.map((s, n) => (
            <button
              key={s.id}
              type="button"
              className="deck-thumb"
              data-current={n === i}
              onClick={() => {
                go(n);
                setGrid(false);
              }}
            >
              <div className="inner" style={{ transform: `scale(${thumbScale})` }}>
                {s.el}
              </div>
              <div className="cap">
                <span>{String(n + 1).padStart(2, "0")}</span>
                {s.nav}
              </div>
            </button>
          ))}
        </div>
      ) : (
        <>
          <div className="deck-viewport">
            <div className="deck-stage" style={{ transform: `scale(${scale})` }}>
              {current?.el}
              {showFooter && (
                <div className="deck-footer">
                  <span>VENTUNO Q · on-device try-on</span>
                  <span className="bar">
                    <i style={{ width: `${((i + 1) / slides.length) * 100}%` }} />
                  </span>
                  <span className="count">
                    {String(i + 1).padStart(2, "0")} / {slides.length}
                  </span>
                </div>
              )}
            </div>
          </div>
          <div className="deck-keys">
            <span>
              <kbd>←</kbd>
              <kbd>→</kbd> navigate
            </span>
            <span>
              <kbd>G</kbd> overview
            </span>
            <span>
              <kbd>F</kbd> full screen
            </span>
          </div>
        </>
      )}
    </div>
  );
}
