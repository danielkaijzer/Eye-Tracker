"use client";

import { useEffect, useRef } from "react";

// Emulator publishes gaze_point in these screen pixels (see
// scripts/extras/gaze_emulator.py defaults). Adjust if you run the
// emulator with custom --width/--height.
const EMULATOR_WIDTH = 1920;
const EMULATOR_HEIGHT = 1080;

// Per-sample blob — small radius, low alpha so heat *accumulates* over
// many samples instead of any single point dominating.
const BLOB_RADIUS = 40;
const BLOB_ALPHA = 0.06;

export default function HeatmapCanvas() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current; // rectangle representing the screen
    const container = containerRef.current;
    if (!canvas || !container) return;
    const ctx = canvas.getContext("2d"); // paintbrush
    if (!ctx) return;

    const syncCanvasSize = () => {
      const rect = container.getBoundingClientRect();
      canvas.width = Math.max(1, Math.floor(rect.width));
      canvas.height = Math.max(1, Math.floor(rect.height));
    };
    syncCanvasSize();
    window.addEventListener("resize", syncCanvasSize);

    const ws = new WebSocket("ws://localhost:9998");
    ws.onmessage = (e) => {
      try {
        const { gaze_point } = JSON.parse(e.data);
        const x = (gaze_point[0] / EMULATOR_WIDTH) * canvas.width;
        const y = (gaze_point[1] / EMULATOR_HEIGHT) * canvas.height;
        const grad = ctx.createRadialGradient(x, y, 0, x, y, BLOB_RADIUS);
        grad.addColorStop(0, `rgba(239, 68, 68, ${BLOB_ALPHA})`);
        grad.addColorStop(1, "rgba(239, 68, 68, 0)");
        ctx.fillStyle = grad;
        ctx.fillRect(
          x - BLOB_RADIUS,
          y - BLOB_RADIUS,
          BLOB_RADIUS * 2,
          BLOB_RADIUS * 2,
        );
      } catch {
        // Nothing happens here because we only hit this catch when we receive a malformed package. We do nothing with malformed packages.
      }
    };

    return () => {
      ws.close();
      window.removeEventListener("resize", syncCanvasSize);
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
        className="relative h-[500px] w-full overflow-hidden rounded-xl border border-zinc-700 bg-black"
      >
        <canvas ref={canvasRef} className="block h-full w-full" />
      </div>
    </div>
  );
}
