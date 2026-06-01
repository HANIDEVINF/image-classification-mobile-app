import type { Metadata } from "next"
import "./globals.css"

export const metadata: Metadata = {
  title: "Keras Vision Classifier",
  description: "A Keras CIFAR-10 image classifier with browser inference.",
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
