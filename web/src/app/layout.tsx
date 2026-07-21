import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AML Detection Studio",
  description: "Evaluasi rule-based, ML anomaly detection, dan simulasi inference AML.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="id">
      <body>{children}</body>
    </html>
  );
}
