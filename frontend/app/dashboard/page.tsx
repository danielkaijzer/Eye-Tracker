"use client";

import { supabase } from "@/lib/supabase";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";

export default function DashboardPage() {
  const router = useRouter();
  const [ready, setReady] = useState(false);

  useEffect(() => {
    let cancelled = false;
    supabase.auth.getSession().then(({ data: { session } }) => {
      if (cancelled) return;
      if (!session) {
        router.replace("/login");
        return;
      }
      setReady(true);
    });
    return () => {
      cancelled = true;
    };
  }, [router]);

  if (!ready) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-[#121212] text-zinc-500">
        Loading…
      </div>
    );
  }

  /** `python scripts/linux_cam_stream.py` → MJPEG at this URL */
  const cameraStreamUrl =
    process.env.NEXT_PUBLIC_CAMERA_STREAM_URL ?? "http://127.0.0.1:5000/";

  return (
    <div className="flex min-h-screen flex-col bg-[#121212] font-sans text-white">
      <header className="mx-4 mt-4 grid shrink-0 grid-cols-[1fr_auto_1fr] items-center gap-4 rounded-2xl border border-white bg-black px-6 py-4">
        <div className="flex h-14 w-14 items-center justify-center rounded-full border border-zinc-600 bg-zinc-950 text-xs text-zinc-400">
          Logo
        </div>
        <nav className="flex justify-center gap-8 text-sm font-medium text-white max-sm:gap-4 max-sm:text-xs">
          <span>HeatMap</span>
          <span>Calibration</span>
          <span>ML Analytics</span>
        </nav>
        <div className="flex justify-end">
          <div
            className="flex h-10 w-10 items-center justify-center rounded-full border border-white bg-zinc-950"
            aria-hidden
          >
            <svg className="h-6 w-6 text-white" viewBox="0 0 24 24" fill="currentColor">
              <path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z" />
            </svg>
          </div>
        </div>
      </header>

      <div className="flex flex-1 flex-col gap-4 p-4 lg:flex-row">
        <section className="flex min-h-[360px] flex-1 flex-col rounded-2xl border border-white p-6 lg:min-h-0">
          <h2 className="mb-2 text-lg font-semibold">Eye Tracker</h2>
          <p className="mb-6 max-w-xl text-sm leading-relaxed text-zinc-300">
            Displays a red circle in relation to where the user is looking on
            the screen here.
          </p>
          <div className="relative min-h-[240px] flex-1 overflow-hidden rounded-xl bg-black">
            {/* eslint-disable-next-line @next/next/no-img-element -- MJPEG stream from Flask; next/image does not support this */}
            <img
              src={cameraStreamUrl}
              alt="Live camera"
              className="h-full min-h-[240px] w-full object-contain"
            />
            <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
              <div className="h-8 w-8 rounded-full border-2 border-red-500 bg-transparent" />
            </div>
          </div>
        </section>

        <aside className="flex w-full shrink-0 flex-col gap-4 lg:w-80">
          <div className="flex min-h-[220px] flex-1 flex-col overflow-hidden rounded-2xl border border-white bg-black/40">
            <div className="border-b border-zinc-700 px-4 py-3 text-sm font-medium">
              Internal Eye Feed
            </div>
            <div className="relative flex flex-1 items-center justify-center bg-black">
              <div className="absolute inset-0 bg-gradient-to-br from-zinc-800 to-black" />
              <span className="relative z-10 text-xs text-zinc-500">
                Eye camera preview
              </span>
            </div>
          </div>

          <div className="rounded-2xl border border-white p-5">
            <h3 className="mb-4 text-sm font-medium">Live Metrics:</h3>
            <ul className="space-y-2 text-sm">
              <li className="flex justify-between gap-4">
                <span className="text-white">vid rate:</span>
                <span className="text-[#4FC3F7]">x Hz</span>
              </li>
              <li className="flex justify-between gap-4">
                <span className="text-white">Gaze X:</span>
                <span className="text-[#4FC3F7]">x coord</span>
              </li>
              <li className="flex justify-between gap-4">
                <span className="text-white">Gaze Y:</span>
                <span className="text-[#4FC3F7]">y coord</span>
              </li>
              <li className="flex justify-between gap-4">
                <span className="text-white">Accuracy:</span>
                <span className="text-[#4FC3F7]">%</span>
              </li>
              <li className="flex justify-between gap-4">
                <span className="text-white">Latency:</span>
                <span className="text-[#4FC3F7]">ms</span>
              </li>
            </ul>
          </div>
        </aside>
      </div>

      <footer className="shrink-0 border-t border-zinc-800 p-4 pb-8">
        <h3 className="mb-3 text-sm font-medium text-white">Session Log:</h3>
        <pre className="font-mono text-sm leading-relaxed text-[#4CAF50]">
          {`> Camera stream initialized
> Handshake with Raspberry Pi successful
> Calibration loaded (Profile: User_01)`}
        </pre>
      </footer>
    </div>
  );
}
