# Eye Tracker — Frontend

Next.js 16 / React 19 dashboard for the Eye Tracker project. Authenticates with Supabase, consumes MJPEG streams from the Python backend, and overlays live gaze (`GazeDot`) and session heatmaps (`HeatmapCanvas`).

See the [root README](../README.md) for the full system overview and the [Figma wireframes](https://www.figma.com/design/WKvgVunFAci4GsTFlWHqsr/Opticore?node-id=0-1&p=f) for the intended UI.

## Install

Requires Node 20+.

```
npm install
cp .env.local.example .env.local    # then fill in the two Supabase keys
```

`.env.local` must define:

```
NEXT_PUBLIC_SUPABASE_URL=...
NEXT_PUBLIC_SUPABASE_ANON_KEY=...
```

Both are public (the anon key is safe to ship to the browser); never put the Supabase service-role key in this file. `lib/supabase.ts` throws at startup if either is missing.

## Run

```
npm run dev          # dev server at http://localhost:3000
npm run build        # production build
npm run start        # serve the production build
npm run lint         # Next.js ESLint preset
```

Open <http://localhost:3000>, sign up or log in, then click **Load Calibration** to reuse the most recent saved fit from the backend.

## Backend dependency

The dashboard expects the Python backend running in web mode in a separate terminal (from the repo root):

```
py -m scripts.eyetracker --web
```

That mode replaces the local cv2 windows with Flask MJPEG endpoints the dashboard pulls from. Without it, the gaze / camera panels will show blank streams.

## Layout

```
app/                # Next.js app-router pages
    dashboard/      # calibration, games, heatmap, ml-analytics, profile
    login/          # Supabase email/password login
    signup/         # Supabase signup
components/         # GazeDot, HeatmapCanvas
lib/supabase.ts     # Supabase browser client
```

## Code style

Next.js ESLint preset (`eslint-config-next/core-web-vitals` + `eslint-config-next/typescript`), configured in `eslint.config.mjs`. `tsconfig.json` enables `strict: true`. Exported React components carry JSDoc.
