"use client";

import { supabase } from "@/lib/supabase";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";

/** `python -m scripts.eyetracker --web` → MJPEG at these URLs */
const eyeStreamUrl =
  process.env.NEXT_PUBLIC_EYE_STREAM_URL ?? "http://127.0.0.1:5001/eye.mjpg";
const sceneStreamUrl =
  process.env.NEXT_PUBLIC_SCENE_STREAM_URL ??
  "http://127.0.0.1:5001/scene.mjpg";
const gazeJsonUrl =
  process.env.NEXT_PUBLIC_GAZE_JSON_URL ?? "http://127.0.0.1:5001/gaze.json";
const loadCalibrationUrl =
  process.env.NEXT_PUBLIC_LOAD_CALIBRATION_URL ?? "http://127.0.0.1:5001/load";

export default function DashboardPage() {
  const router = useRouter();
  const [ready, setReady] = useState(false);
  const [gaze, setGaze] = useState<{ x: number | null; y: number | null }>({
    x: null,
    y: null,
  });

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

  useEffect(() => {
    if (!ready) return;
    let cancelled = false;
    const tick = async () => {
      try {
        const res = await fetch(gazeJsonUrl, { cache: "no-store" });
        if (!res.ok) return;
        const data = (await res.json()) as { x: number | null; y: number | null };
        if (!cancelled) setGaze(data);
      } catch {
        // Backend probably not running; leave previous value.
      }
    };
    const id = setInterval(tick, 100);
    tick();
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [ready]);

  if (!ready) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-[#121212] text-zinc-500">
        Loading…
      </div>
    );
  }

  const handleLoadCalibration = async () => {
    try {
      await fetch(loadCalibrationUrl);
    } catch (err) {
      console.error("Failed to request calibration load:", err);
    }
  };

  return (
    <div className="flex min-h-screen flex-col bg-[#121212] font-sans text-white">
      <header className="mx-4 mt-4 grid shrink-0 grid-cols-[1fr_auto_1fr] items-center gap-4 rounded-2xl border border-white bg-black px-6 py-4">
        <Link
          href="/dashboard"
          aria-label="Go back to dashboard"
          className="flex h-14 w-14 items-center justify-center rounded-full border border-zinc-600 bg-zinc-950 text-xs text-zinc-400 transition-colors hover:border-blue-400 hover:text-blue-400"
        >
          Logo
        </Link>
        <nav className="flex justify-center gap-8 text-sm font-medium text-white max-sm:gap-4 max-sm:text-xs">
          <Link
            href="/dashboard/heatmap"
            className="hover:text-blue-400 transition-colors"
          >
            HeatMap
          </Link>
          <Link
            href="/dashboard/calibration"
            className="hover:text-blue-400 transition-colors"
          >
            Calibration
          </Link>
          <Link
            href="/dashboard/ml-analytics"
            className="hover:text-blue-400 transition-colors"
          >
            ML Analytics
          </Link>
          <Link
            href="/dashboard/games"
            className="hover:text-blue-400 transition-colors"
          >
            Games
          </Link>
        </nav>
        <div className="flex justify-end">
          <Link
            href="/dashboard/profile"
            aria-label="Open profile page"
            className="flex h-10 w-10 items-center justify-center rounded-full border border-white bg-zinc-950 transition-colors hover:border-blue-400 hover:text-blue-400"
          >
            <svg
              className="h-6 w-6 text-white"
              viewBox="0 0 24 24"
              fill="currentColor"
            >
              <path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z" />
            </svg>
          </Link>
        </div>
      </header>

      <div className="flex flex-1 flex-col gap-4 p-4 lg:flex-row">
        <section className="flex min-h-[360px] flex-1 flex-col rounded-2xl border border-white p-6 lg:min-h-0">
          <div className="mb-2 flex items-center justify-between gap-4">
            <h2 className="text-lg font-semibold">Eye Tracker</h2>
            <button
              type="button"
              onClick={handleLoadCalibration}
              className="rounded-full border border-white bg-black px-4 py-1.5 text-xs font-medium text-white hover:bg-zinc-900"
            >
              Load Calibration
            </button>
          </div>
          <p className="mb-6 max-w-xl text-sm leading-relaxed text-zinc-300">
            Displays a red circle in relation to where the user is looking on
            the screen here.
          </p>
          <div className="relative mx-auto aspect-[4/3] w-full max-w-[640px] overflow-hidden rounded-xl bg-black">
            {/* eslint-disable-next-line @next/next/no-img-element -- MJPEG stream from Flask; next/image does not support this */}
            <img
              src={sceneStreamUrl}
              alt="Scene camera with gaze overlay"
              className="h-full w-full object-contain"
            />
          </div>
        </section>

        <aside className="flex w-full shrink-0 flex-col gap-4 lg:w-80">
          <div className="flex min-h-[220px] flex-1 flex-col overflow-hidden rounded-2xl border border-white bg-black/40">
            <div className="border-b border-zinc-700 px-4 py-3 text-sm font-medium">
              Internal Eye Feed
            </div>
            <div className="relative flex flex-1 items-center justify-center bg-black">
              {/* eslint-disable-next-line @next/next/no-img-element -- MJPEG stream from Flask; next/image does not support this */}
              <img
                src={eyeStreamUrl}
                alt="Eye camera with pupil detection overlay"
                className="h-full w-full object-contain"
              />
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
                <span className="text-[#4FC3F7]">
                  {gaze.x ?? "—"}
                </span>
              </li>
              <li className="flex justify-between gap-4">
                <span className="text-white">Gaze Y:</span>
                <span className="text-[#4FC3F7]">
                  {gaze.y ?? "—"}
                </span>
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
> Handshake with Jetson successful
> Calibration loaded (Profile: User_01)`}
        </pre>
      </footer>
    </div>
  );
}
