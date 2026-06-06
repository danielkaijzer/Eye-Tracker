"use client";

import { supabase } from "@/lib/supabase";
import { useRouter } from "next/navigation";
import { FormEvent, useState } from "react";

export function SignupForm() {
  const router = useRouter();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [info, setInfo] = useState<string | null>(null);

  async function handleSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setError(null);
    setInfo(null);
    const form = e.currentTarget;
    const fd = new FormData(form);
    const email = String(fd.get("email") ?? "").trim();
    const password = String(fd.get("password") ?? "");
    const confirm = String(fd.get("confirmPassword") ?? "");

    if (password !== confirm) {
      setError("Passwords do not match.");
      return;
    }

    setLoading(true);
    const { data, error: signUpError } = await supabase.auth.signUp({
      email,
      password,
      options: {
        emailRedirectTo: `${typeof window !== "undefined" ? window.location.origin : ""}/dashboard`,
      },
    });
    setLoading(false);

    if (signUpError) {
      setError(signUpError.message);
      return;
    }

    if (data.user && !data.session) {
      setInfo("Check your email for a confirmation link before signing in.");
      return;
    }

    router.push("/dashboard");
    router.refresh();
  }

  return (
    <form
      onSubmit={handleSubmit}
      className="flex w-full max-w-sm flex-col gap-5"
    >
      {error ? (
        <p
          className="rounded-lg border border-red-500/50 bg-red-950/40 px-3 py-2 text-sm text-red-200"
          role="alert"
        >
          {error}
        </p>
      ) : null}
      {info ? (
        <p
          className="rounded-lg border border-emerald-500/50 bg-emerald-950/40 px-3 py-2 text-sm text-emerald-100"
          role="status"
        >
          {info}
        </p>
      ) : null}

      <div className="flex flex-col gap-2 text-left">
        <label htmlFor="signup-email" className="text-sm text-zinc-400">
          Email
        </label>
        <input
          id="signup-email"
          name="email"
          type="email"
          autoComplete="email"
          required
          disabled={loading}
          className="rounded-lg border border-zinc-600 bg-zinc-900 px-4 py-3 text-white outline-none ring-white/20 placeholder:text-zinc-500 focus:border-white focus:ring-2 disabled:opacity-50"
          placeholder="you@example.com"
        />
      </div>
      <div className="flex flex-col gap-2 text-left">
        <label htmlFor="signup-password" className="text-sm text-zinc-400">
          Password
        </label>
        <input
          id="signup-password"
          name="password"
          type="password"
          autoComplete="new-password"
          required
          minLength={8}
          disabled={loading}
          className="rounded-lg border border-zinc-600 bg-zinc-900 px-4 py-3 text-white outline-none ring-white/20 placeholder:text-zinc-500 focus:border-white focus:ring-2 disabled:opacity-50"
          placeholder="••••••••"
        />
      </div>
      <div className="flex flex-col gap-2 text-left">
        <label htmlFor="signup-confirm" className="text-sm text-zinc-400">
          Confirm password
        </label>
        <input
          id="signup-confirm"
          name="confirmPassword"
          type="password"
          autoComplete="new-password"
          required
          minLength={8}
          disabled={loading}
          className="rounded-lg border border-zinc-600 bg-zinc-900 px-4 py-3 text-white outline-none ring-white/20 placeholder:text-zinc-500 focus:border-white focus:ring-2 disabled:opacity-50"
          placeholder="••••••••"
        />
      </div>
      <button
        type="submit"
        disabled={loading}
        className="mt-2 w-full rounded-full border border-white bg-black py-3 text-base font-medium text-white disabled:cursor-not-allowed disabled:opacity-60"
      >
        {loading ? "Creating account…" : "Create account"}
      </button>
    </form>
  );
}
