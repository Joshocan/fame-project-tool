import type { Metadata } from "next";
import { AppShell } from "@/components/AppShell";
import { appConfig } from "@/lib/config";
import { listContainers } from "@/lib/storage";
import "./globals.css";

export const metadata: Metadata = {
  title: "FAME Web",
  description: "Lightweight web demo for artefact-driven model construction experiments."
};

export default async function RootLayout({
  children
}: Readonly<{ children: React.ReactNode }>) {
  const containers = await listContainers();

  return (
    <html lang="en">
      <body>
        <AppShell config={appConfig} containers={containers}>
          {children}
        </AppShell>
      </body>
    </html>
  );
}
