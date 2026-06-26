import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "River — Image Exploration",
  description: "Lateral visual exploration of image spaces in SigLIP ViT-B/16 (384px)",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
