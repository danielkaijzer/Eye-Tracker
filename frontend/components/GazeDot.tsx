"use client";

import { useEffect, useState } from "react";

export default function GazeDot() {
  const [pt, setPt] = useState<[number, number] | null>(null);

  useEffect(() => {
    const ws = new WebSocket("ws://localhost:9998");
    ws.onmessage = (e) => setPt(JSON.parse(e.data).gaze_point);
    return () => ws.close();
  }, []);

  if (!pt) return null;

  return (
    <div
      style={{
        position: "fixed",
        left: pt[0],
        top: pt[1],
        width: 16,
        height: 16,
        borderRadius: 999,
        background: "red",
        pointerEvents: "none",
      }}
    />
  );
}
