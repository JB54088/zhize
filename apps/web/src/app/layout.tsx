import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "知择 · 让判断有据可依",
  description: "从可信来源中梳理证据，帮助你看清问题，再做选择。",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}
