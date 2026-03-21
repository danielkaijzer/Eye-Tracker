import Link from "next/link";
import { LoginForm } from "./login-form";

export const metadata = {
  title: "Log in | Eye Tracker",
  description: "Sign in to your Eye Tracker account.",
};

export default function LoginPage() {
  return (
    <div className="min-h-screen bg-[#1a1a1a] flex flex-col items-center px-6 py-12">
      <div className="flex w-full max-w-md flex-col items-center">
        <Link
          href="/"
          className="mb-10 self-start text-sm text-zinc-400 transition-colors hover:text-white"
        >
          ← Back
        </Link>
        <h1 className="mb-2 text-2xl font-semibold tracking-tight text-white">
          Log in
        </h1>
        <p className="mb-10 text-center text-sm text-zinc-400">
          Enter your credentials to continue.
        </p>
        <LoginForm />
        <p className="mt-8 text-center text-sm text-zinc-500">
          Don&apos;t have an account?{" "}
          <Link href="/signup" className="text-white underline-offset-4 hover:underline">
            Sign up
          </Link>
        </p>
      </div>
    </div>
  );
}
