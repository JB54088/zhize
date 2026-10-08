import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "知择 · AI 信息决策",
  description: "基于可追溯证据，帮助你更清晰地判断信息与选择。",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}

