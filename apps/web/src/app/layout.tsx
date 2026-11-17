import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "知择 · 信息万千，知而善择",
  description: "面向日常与专业场景的信息搜索和决策底座。",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}
