import Link from "next/link";
import { SignupForm } from "./signup-form";

export const metadata = {
  title: "Sign up | Eye Tracker",
  description: "Create an Eye Tracker account.",
};

export default function SignupPage() {
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
          Sign up
        </h1>
        <p className="mb-10 text-center text-sm text-zinc-400">
          Create an account to get started.
        </p>
        <SignupForm />
        <p className="mt-8 text-center text-sm text-zinc-500">
          Already have an account?{" "}
          <Link href="/login" className="text-white underline-offset-4 hover:underline">
            Log in
          </Link>
        </p>
      </div>
    </div>
  );
}
