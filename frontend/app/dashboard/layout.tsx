import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Dashboard | Eye Tracker",
  description: "Eye tracking dashboard",
};

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return children;
}
