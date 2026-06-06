"use client";

import { useEffect, useRef, useState } from "react";

const gazeJsonUrl = process.env.NEXT_PUBLIC_GAZE_JSON_URL ?? "/gaze.json";
const sceneStreamUrl = process.env.NEXT_PUBLIC_SCENE_STREAM_URL ?? "/scene.mjpg";

const BLOB_RADIUS = 50;
const BLOB_ALPHA = 0.18;
const POLL_MS = 100;

type GazeSample = {
  x: number | null;
  y: number | null;
  w: number | null;
  h: number | null;
};

/**
 * Scene-camera MJPEG stream with a cumulative gaze heatmap overlaid.
 *
 * Polls `NEXT_PUBLIC_GAZE_JSON_URL` (default `/gaze.json`) every 100ms and
 * stamps a translucent radial blob at each new gaze sample on a canvas
 * positioned over `NEXT_PUBLIC_SCENE_STREAM_URL`. The container aspect
 * ratio follows the scene-cam dimensions reported by the backend so the
 * heat stays aligned with the underlying image. Stationary repeats are
 * skipped so a held gaze doesn't burn a single hot spot. The "Clear"
 * button wipes the canvas.
 *
 * Takes no props. Cleans up the poll interval and ResizeObserver on
 * unmount; silently no-ops when the backend is unreachable.
 */
export default function HeatmapCanvas() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  // Container aspect ratio tracks scene-cam dimensions from /gaze.json so
  // image and canvas share the exact same rect; otherwise heat drifts off-frame.
  const [aspect, setAspect] = useState<string>("4 / 3");

  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const syncCanvasSize = () => {
      const rect = container.getBoundingClientRect();
      canvas.width = Math.max(1, Math.floor(rect.width));
      canvas.height = Math.max(1, Math.floor(rect.height));
    };
    syncCanvasSize();
    const observer = new ResizeObserver(syncCanvasSize);
    observer.observe(container);

    let cancelled = false;
    let lastX: number | null = null;
    let lastY: number | null = null;

    let lastAspect = "";
    const paint = (data: GazeSample) => {
      if (!data.w || !data.h) return;
      const nextAspect = `${data.w} / ${data.h}`;
      if (nextAspect !== lastAspect) {
        lastAspect = nextAspect;
        setAspect(nextAspect);
      }
      if (data.x == null || data.y == null) return;
      // Skip stationary repeats so a still gaze doesn't burn a hot spot.
      if (data.x === lastX && data.y === lastY) return;
      lastX = data.x;
      lastY = data.y;
      const cx = (data.x / data.w) * canvas.width;
      const cy = (data.y / data.h) * canvas.height;
      const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, BLOB_RADIUS);
      grad.addColorStop(0, `rgba(239, 68, 68, ${BLOB_ALPHA})`);
      grad.addColorStop(1, "rgba(239, 68, 68, 0)");
      ctx.fillStyle = grad;
      ctx.fillRect(
        cx - BLOB_RADIUS,
        cy - BLOB_RADIUS,
        BLOB_RADIUS * 2,
        BLOB_RADIUS * 2,
      );
    };

    const tick = async () => {
      try {
        const res = await fetch(gazeJsonUrl, { cache: "no-store" });
        if (!res.ok) return;
        const data = (await res.json()) as GazeSample;
        if (!cancelled) paint(data);
      } catch {
        // Backend probably not running; leave previous heat in place.
      }
    };
    const id = setInterval(tick, POLL_MS);
    tick();

    return () => {
      cancelled = true;
      clearInterval(id);
      observer.disconnect();
    };
  }, []);

  const clearHeatmap = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    ctx?.clearRect(0, 0, canvas.width, canvas.height);
  };

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between gap-4">
        <button
          type="button"
          onClick={clearHeatmap}
          className="shrink-0 rounded-full border border-zinc-700 bg-zinc-950 px-4 py-1.5 text-xs font-medium text-white hover:bg-zinc-900"
        >
          Clear
        </button>
      </div>
      <div
        ref={containerRef}
        style={{ aspectRatio: aspect }}
        className="relative mx-auto w-full max-w-[960px] overflow-hidden rounded-xl border border-zinc-700 bg-black"
      >
        {/* eslint-disable-next-line @next/next/no-img-element -- MJPEG stream from Flask; next/image does not support this */}
        <img
          src={sceneStreamUrl}
          alt="Scene camera"
          className="absolute inset-0 h-full w-full"
        />
        <canvas
          ref={canvasRef}
          className="absolute inset-0 block h-full w-full"
        />
      </div>
    </div>
  );
}
