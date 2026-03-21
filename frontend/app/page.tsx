import Link from "next/link";

export default function Home() {
  return (
    <div className="min-h-screen bg-[#1a1a1a] flex flex-col items-center justify-center px-6 py-12">
      <div className="flex w-full max-w-md flex-col items-center text-center">
        <h1 className="mb-3 text-3xl font-semibold tracking-tight text-white">
          Eye Tracker
        </h1>
        <p className="mb-12 max-w-sm text-base leading-relaxed text-zinc-400">
          Sign in to your account or create one to get started.
        </p>

        <div className="flex w-full max-w-xs flex-col gap-4 sm:max-w-sm">
          <Link
            href="/login"
            className="w-full rounded-full border border-white bg-black py-3 text-center text-base font-medium text-white"
          >
            Log in
          </Link>
          <Link
            href="/signup"
            className="w-full rounded-full border border-white bg-black py-3 text-center text-base font-medium text-white"
          >
            Sign up
          </Link>
        </div>
      </div>
    </div>
  );
}
