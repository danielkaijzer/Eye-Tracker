"use client";

import { supabase } from "@/lib/supabase";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";

export default function CalibrationPage() {
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

    return (
        <div className="flex min-h-screen flex-col bg-[#121212] font-sans text-white">
            <header className="mx-4 mt-4 grid shrink-0 grid-cols-[1fr_auto_1fr] items-center gap-4 rounded-2xl border border-white bg-black px-6 py-4">
                <div className="flex h-14 w-14 items-center justify-center rounded-full border border-zinc-600 bg-zinc-950 text-xs text-zinc-400">
                    Logo
                </div>
                <nav className="flex justify-center gap-8 text-sm font-medium text-white max-sm:gap-4 max-sm:text-xs">
                    <Link
                        href="/dashboard/heatmap"
                        className="hover:text-blue-400 transition-colors"
                    >
                        HeatMap
                    </Link>
                    <Link
                        href="/dashboard/calibration"
                        className="hover:text-blue-400 transition-colors border-b-2 border-blue-400"
                    >
                        Calibration
                    </Link>
                    <Link
                        href="/dashboard/ml-analytics"
                        className="hover:text-blue-400 transition-colors"
                    >
                        ML Analytics
                    </Link>
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

            <div className="flex flex-1 flex-col gap-4 p-4">
                <div className="flex items-center justify-between">
                    <h1 className="text-3xl font-bold">Calibration</h1>
                    <Link
                        href="/dashboard"
                        className="px-4 py-2 rounded-lg border border-zinc-600 bg-zinc-950 text-sm font-medium hover:bg-zinc-900 transition-colors"
                    >
                        Back to Dashboard
                    </Link>
                </div>

                <section className="flex-1 min-h-[600px] rounded-2xl border border-white p-6">
                    <h2 className="mb-4 text-lg font-semibold">Eye Tracker Calibration</h2>
                    <p className="mb-6 text-sm text-zinc-300">
                        Calibrate the eye tracker by following on-screen instructions to ensure accurate tracking.
                    </p>
                    <div className="w-full h-[500px] rounded-xl bg-black border border-zinc-700 flex items-center justify-center">
                        <p className="text-zinc-500">Calibration interface will be implemented here</p>
                    </div>
                </section>
            </div>
        </div>
    );
}
