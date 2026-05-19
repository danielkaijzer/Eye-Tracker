import type { NextConfig } from "next";

const FLASK_ORIGIN = process.env.FLASK_ORIGIN ?? "http://127.0.0.1:5001";

const nextConfig: NextConfig = {
  allowedDevOrigins: [
    '10.170.39.115',
    '*.trycloudflare.com',
    'brand-property-knife-instructors.trycloudflare.com',
  ],
  async rewrites() {
    return [
      { source: "/eye.mjpg", destination: `${FLASK_ORIGIN}/eye.mjpg` },
      { source: "/scene.mjpg", destination: `${FLASK_ORIGIN}/scene.mjpg` },
      { source: "/gaze.json", destination: `${FLASK_ORIGIN}/gaze.json` },
      { source: "/load", destination: `${FLASK_ORIGIN}/load` },
    ];
  },
};

export default nextConfig;