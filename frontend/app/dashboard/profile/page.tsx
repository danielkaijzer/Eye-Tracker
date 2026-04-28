"use client";

import { supabase } from "@/lib/supabase";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useMemo, useState } from "react";

function formatDuration(totalSeconds: number) {
    const hours = Math.floor(totalSeconds / 3600);
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    const seconds = totalSeconds % 60;
    return [hours, minutes, seconds]
        .map((value) => String(value).padStart(2, "0"))
        .join(":");
}

export default function ProfilePage() {
    const router = useRouter();
    const [ready, setReady] = useState(false);
    const [loadingSignOut, setLoadingSignOut] = useState(false);
    const [sessionSeconds, setSessionSeconds] = useState(0);
    const [email, setEmail] = useState<string>("");
    const [username, setUsername] = useState<string>("");

    useEffect(() => {
        let cancelled = false;

        supabase.auth.getSession().then(({ data: { session } }) => {
            if (cancelled) return;

            if (!session) {
                router.replace("/login");
                return;
            }

            const user = session.user;
            const metadataUsername =
                typeof user.user_metadata?.username === "string"
                    ? user.user_metadata.username
                    : typeof user.user_metadata?.full_name === "string"
                        ? user.user_metadata.full_name
                        : "";

            setEmail(user.email ?? "");
            setUsername(metadataUsername || user.email?.split("@")[0] || "User");
            setReady(true);
        });

        const startedAt = Date.now();
        const interval = window.setInterval(() => {
            setSessionSeconds(Math.floor((Date.now() - startedAt) / 1000));
        }, 1000);

        return () => {
            cancelled = true;
            window.clearInterval(interval);
        };
    }, [router]);

    const maskedPassword = useMemo(() => "••••••••", []);

    async function handleLogout() {
        setLoadingSignOut(true);
        await supabase.auth.signOut();
        router.replace("/login");
        router.refresh();
    }

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
                <Link
                    href="/dashboard"
                    aria-label="Go back to dashboard"
                    className="flex h-14 w-14 items-center justify-center rounded-full border border-zinc-600 bg-zinc-950 text-xs text-zinc-400 transition-colors hover:border-blue-400 hover:text-blue-400"
                >
                    Logo
                </Link>
                <div className="text-center text-sm font-medium uppercase tracking-[0.3em] text-zinc-400">
                    Profile
                </div>
                <div className="flex justify-end">
                    <div className="flex h-10 w-10 items-center justify-center rounded-full border border-blue-400 bg-zinc-950 text-blue-300">
                        <svg className="h-6 w-6" viewBox="0 0 24 24" fill="currentColor">
                            <path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z" />
                        </svg>
                    </div>
                </div>
            </header>

            <main className="mx-auto flex w-full max-w-5xl flex-1 flex-col gap-6 px-4 py-6 lg:flex-row">
                <section className="flex-1 rounded-3xl border border-white bg-black/40 p-6 shadow-2xl shadow-black/30">
                    <div className="flex flex-col gap-6 sm:flex-row sm:items-center sm:justify-between">
                        <div>
                            <p className="text-sm uppercase tracking-[0.3em] text-zinc-500">Account</p>
                            <h1 className="mt-2 text-3xl font-semibold">{username}</h1>
                            <p className="mt-2 text-sm text-zinc-400">Manage your identity and session access.</p>
                        </div>
                        <div className="flex h-24 w-24 items-center justify-center rounded-full border border-zinc-700 bg-zinc-950 text-3xl font-semibold text-white">
                            {username.slice(0, 1).toUpperCase()}
                        </div>
                    </div>

                    <div className="mt-8 grid gap-4 md:grid-cols-2">
                        <div className="rounded-2xl border border-zinc-800 bg-zinc-950/80 p-4">
                            <p className="text-xs uppercase tracking-[0.25em] text-zinc-500">Username</p>
                            <p className="mt-2 text-lg text-white">{username}</p>
                        </div>
                        <div className="rounded-2xl border border-zinc-800 bg-zinc-950/80 p-4">
                            <p className="text-xs uppercase tracking-[0.25em] text-zinc-500">Email</p>
                            <p className="mt-2 text-lg text-white break-all">{email || "Not available"}</p>
                        </div>
                        <div className="rounded-2xl border border-zinc-800 bg-zinc-950/80 p-4">
                            <p className="text-xs uppercase tracking-[0.25em] text-zinc-500">Password</p>
                            <p className="mt-2 text-lg text-white">{maskedPassword}</p>
                            <p className="mt-2 text-sm text-zinc-500">Passwords are not readable from Supabase auth. Use sign-in or reset flows to change it.</p>
                        </div>
                        <div className="rounded-2xl border border-zinc-800 bg-zinc-950/80 p-4">
                            <p className="text-xs uppercase tracking-[0.25em] text-zinc-500">Usage Time</p>
                            <p className="mt-2 text-lg text-white">{formatDuration(sessionSeconds)}</p>
                            <p className="mt-2 text-sm text-zinc-500">Current session duration since this page opened.</p>
                        </div>
                    </div>
                </section>

                <aside className="w-full shrink-0 space-y-4 lg:w-80">
                    <div className="rounded-3xl border border-white bg-black/40 p-6">
                        <h2 className="text-lg font-semibold">Profile actions</h2>
                        <p className="mt-2 text-sm leading-relaxed text-zinc-400">
                            Use the dashboard controls below to leave your account securely.
                        </p>
                        <button
                            type="button"
                            onClick={handleLogout}
                            disabled={loadingSignOut}
                            className="mt-6 w-full rounded-full border border-white bg-black px-4 py-3 text-sm font-medium text-white transition-colors hover:bg-zinc-900 disabled:cursor-not-allowed disabled:opacity-60"
                        >
                            {loadingSignOut ? "Logging out…" : "Log out"}
                        </button>
                    </div>

                    <div className="rounded-3xl border border-white bg-black/40 p-6">
                        <h2 className="text-lg font-semibold">Account notes</h2>
                        <ul className="mt-4 space-y-3 text-sm text-zinc-400">
                            <li>Username falls back to the Supabase metadata name or email prefix.</li>
                            <li>Email is pulled from the current authenticated session.</li>
                            <li>Password is intentionally masked for security.</li>
                        </ul>
                    </div>
                </aside>
            </main>
        </div>
    );
}