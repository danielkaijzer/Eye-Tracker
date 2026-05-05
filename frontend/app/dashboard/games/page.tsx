"use client";

import { supabase } from "@/lib/supabase";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useMemo, useRef, useState } from "react";

const GAME_DURATION_SECONDS = 30;
const CELL_SIZE = 60; // Minimum size of each cell in pixels
const PREVIEW_GRID_SIZE = 4; // Fixed 4x4 preview grid
const PREVIEW_CELLS = PREVIEW_GRID_SIZE * PREVIEW_GRID_SIZE;

function createTarget(previousTarget: number, cellCount: number) {
  let nextTarget = previousTarget;
  while (nextTarget === previousTarget) {
    nextTarget = Math.floor(Math.random() * cellCount);
  }
  return nextTarget;
}

function calculateGridSize(width: number, height: number): number {
  // Calculate how many cells fit in each dimension
  const cellsX = Math.floor(width / CELL_SIZE);
  const cellsY = Math.floor(height / CELL_SIZE);

  // Use the smaller dimension to maintain a square grid
  const gridDim = Math.min(cellsX, cellsY);

  // Clamp between minimum 4x4 and maximum 20x20
  return Math.max(4, Math.min(gridDim, 20));
}

export default function GamesPage() {
  const router = useRouter();
  const gameRef = useRef<HTMLDivElement | null>(null);
  const [ready, setReady] = useState(false);
  const [started, setStarted] = useState(false);
  const [score, setScore] = useState(0);
  const [bestScore, setBestScore] = useState(0);
  const [timeLeft, setTimeLeft] = useState(GAME_DURATION_SECONDS);
  const [targetIndex, setTargetIndex] = useState(0);
  const [statusMessage, setStatusMessage] = useState(
    "Press start to begin the grid challenge.",
  );
  const [gridSize, setGridSize] = useState(PREVIEW_GRID_SIZE);

  const scoreRef = useRef(score);
  scoreRef.current = score;

  const CELLS = gridSize * gridSize;

  useEffect(() => {
    let cancelled = false;

    supabase.auth.getSession().then(({ data: { session } }) => {
      if (cancelled) return;

      if (!session) {
        router.replace("/login");
        return;
      }

      setTargetIndex(Math.floor(Math.random() * CELLS));
      setReady(true);
    });

    return () => {
      cancelled = true;
    };
  }, [router, CELLS]);

  useEffect(() => {
    function handleResize() {
      if (document.fullscreenElement && gameRef.current) {
        const newGridSize = calculateGridSize(
          window.innerWidth,
          window.innerHeight,
        );
        setGridSize(newGridSize);
      }
    }

    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape" && started) {
        e.preventDefault();
        handleExit();
      }
    }

    function handleFullscreenChange() {
      if (started && !document.fullscreenElement) {
        handleExit();
      }
    }

    if (started) {
      document.addEventListener("keydown", handleKeyDown);
      document.addEventListener("fullscreenchange", handleFullscreenChange);
    }

    return () => {
      document.removeEventListener("keydown", handleKeyDown);
      document.removeEventListener("fullscreenchange", handleFullscreenChange);
    };
  }, [started]);

  useEffect(() => {
    if (!started) return;

    const timer = window.setInterval(() => {
      setTimeLeft((currentTime) => {
        if (currentTime <= 1) {
          window.clearInterval(timer);
          const finalScore = scoreRef.current;
          setBestScore((currentBest) => Math.max(currentBest, finalScore));
          setStatusMessage(`Game over. Final score: ${finalScore}.`);
          setStarted(false);

          if (document.fullscreenElement) {
            void document.exitFullscreen().catch(() => undefined);
          }

          return 0;
        }

        return currentTime - 1;
      });
    }, 1000);

    return () => window.clearInterval(timer);
  }, [started]);

  const gridCells = useMemo(
    () => Array.from({ length: gridSize * gridSize }, (_, index) => index),
    [gridSize],
  );

  const previewCells = useMemo(
    () => Array.from({ length: PREVIEW_CELLS }, (_, index) => index),
    [],
  );

  async function handleStart() {
    if (!gameRef.current) return;

    const newGridSize = calculateGridSize(
      window.innerWidth,
      window.innerHeight,
    );

    try {
      if (document.fullscreenElement !== gameRef.current) {
        await gameRef.current.requestFullscreen();
      }
    } catch {
      // Ignore fullscreen errors and still allow the game to run inline.
    }

    setGridSize(newGridSize);
    setScore(0);
    setTimeLeft(GAME_DURATION_SECONDS);
    setTargetIndex(Math.floor(Math.random() * newGridSize * newGridSize));
    setStatusMessage("Find and click the highlighted cell.");
    setStarted(true);
  }

  function handleCellClick(index: number) {
    if (!started || index !== targetIndex) return;

    setScore((currentScore) => currentScore + 1);
    setTargetIndex((previousTarget) =>
      createTarget(previousTarget, gridSize * gridSize),
    );
    setStatusMessage("Good hit. Keep going.");
  }

  async function handleExit() {
    const finalScore = scoreRef.current;
    setStarted(false);
    setTimeLeft(0);
    setBestScore((currentBest) => Math.max(currentBest, finalScore));
    setStatusMessage(`Game ended. Final score: ${finalScore}.`);

    if (document.fullscreenElement) {
      await document.exitFullscreen().catch(() => undefined);
    }
  }

  if (!ready) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-[#121212] text-zinc-500">
        Loading…
      </div>
    );
  }

  if (started) {
    return (
      <div
        ref={gameRef}
        className="fixed inset-0 z-50 flex h-screen w-screen flex-col overflow-hidden bg-black font-sans text-white"
      >
        <div
          className="grid h-full w-full gap-0 bg-black"
          style={{
            gridTemplateColumns: `repeat(${gridSize}, 1fr)`,
            gridTemplateRows: `repeat(${gridSize}, 1fr)`,
          }}
        >
          {gridCells.map((index) => {
            const isTarget = index === targetIndex;

            return (
              <button
                key={index}
                type="button"
                onClick={() => handleCellClick(index)}
                className={`rounded-none border border-zinc-800 p-0 transition-colors ${
                  isTarget ? "bg-emerald-400" : "bg-black hover:bg-zinc-900"
                }`}
                aria-label={isTarget ? "Target cell" : `Cell ${index + 1}`}
              />
            );
          })}
        </div>
      </div>
    );
  }

  return (
    <div
      ref={gameRef}
      className="flex min-h-screen flex-col bg-[#090909] font-sans text-white"
    >
      <header className="flex items-center justify-between border-b border-white/10 px-4 py-4 sm:px-6">
        <Link
          href="/dashboard"
          className="rounded-full border border-zinc-700 bg-zinc-950 px-4 py-2 text-sm text-zinc-300 transition-colors hover:border-blue-400 hover:text-blue-400"
        >
          Back to Dashboard
        </Link>
        <div className="text-center">
          <p className="text-xs uppercase tracking-[0.4em] text-zinc-500">
            Games
          </p>
          <h1 className="text-lg font-semibold">Grid Click Challenge</h1>
        </div>
        <div className="w-24" />
      </header>

      <main className="flex flex-1 items-center justify-center p-4 sm:p-6">
        <div className="grid w-full max-w-5xl gap-6 lg:grid-cols-[320px_1fr]">
          <section className="rounded-3xl border border-white/10 bg-white/5 p-6 shadow-2xl shadow-black/30">
            <p className="text-xs uppercase tracking-[0.35em] text-zinc-500">
              Control Panel
            </p>
            <h2 className="mt-2 text-3xl font-semibold">Stay sharp.</h2>
            <p className="mt-4 text-sm leading-relaxed text-zinc-400">
              Click the highlighted cell as fast as you can. Press Start to
              begin.
            </p>

            <div className="mt-6 grid grid-cols-3 gap-3 text-center">
              <div className="rounded-2xl border border-zinc-800 bg-zinc-950/80 p-3">
                <p className="text-[11px] uppercase tracking-[0.2em] text-zinc-500">
                  Score
                </p>
                <p className="mt-1 text-2xl font-semibold">{score}</p>
              </div>
              <div className="rounded-2xl border border-zinc-800 bg-zinc-950/80 p-3">
                <p className="text-[11px] uppercase tracking-[0.2em] text-zinc-500">
                  Best
                </p>
                <p className="mt-1 text-2xl font-semibold">{bestScore}</p>
              </div>
              <div className="rounded-2xl border border-zinc-800 bg-zinc-950/80 p-3">
                <p className="text-[11px] uppercase tracking-[0.2em] text-zinc-500">
                  Time
                </p>
                <p className="mt-1 text-2xl font-semibold">{timeLeft}s</p>
              </div>
            </div>

            <p className="mt-6 rounded-2xl border border-zinc-800 bg-zinc-950/80 px-4 py-3 text-sm text-zinc-300">
              {statusMessage}
            </p>

            <button
              type="button"
              onClick={started ? handleExit : handleStart}
              className="mt-6 w-full rounded-full bg-white px-4 py-3 text-sm font-semibold text-black transition-transform hover:scale-[1.02]"
            >
              {started ? "End Game" : "Start"}
            </button>
          </section>

          <section className="flex items-center justify-center rounded-3xl border border-white/10 bg-[radial-gradient(circle_at_top,_rgba(79,195,247,0.12),_transparent_40%),linear-gradient(180deg,_rgba(255,255,255,0.03),_rgba(255,255,255,0.01))] p-4 sm:p-6">
            <div className="w-full max-w-4xl">
              <div className="grid grid-cols-4 gap-3 sm:gap-4">
                {previewCells.map((index) => {
                  const isTarget = started && index === targetIndex;

                  return (
                    <button
                      key={index}
                      type="button"
                      onClick={() => handleCellClick(index)}
                      disabled={!started}
                      className={`aspect-square rounded-2xl border transition-all duration-150 ${
                        isTarget
                          ? "border-emerald-400 bg-emerald-400/90 shadow-[0_0_24px_rgba(52,211,153,0.45)]"
                          : "border-white/10 bg-black/40 hover:border-blue-400/60 hover:bg-white/5"
                      } disabled:cursor-default`}
                      aria-label={
                        isTarget ? "Target cell" : `Cell ${index + 1}`
                      }
                    />
                  );
                })}
              </div>
            </div>
          </section>
        </div>
      </main>
    </div>
  );
}
