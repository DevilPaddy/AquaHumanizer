/*
 * AquaHumanizer Pro - App Layout
 * Copyright (c) 2026 AquilaStudios
 * Licensed under the MIT License
 */

import type { Metadata, Viewport } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'

const inter = Inter({ 
  subsets: ['latin'],
  display: 'swap',
  variable: '--font-inter',
})

export const metadata: Metadata = {
  title: 'AquaHumanizer Pro - Professional AI Text Enhancement',
  description: 'Transform AI-generated content with professional styles, ATS optimization, and bullet-point formatting for resumes and business documents.',
  keywords: 'AI text humanizer, resume optimization, ATS friendly, professional writing, text enhancement',
  authors: [{ name: 'AquaHumanizer Team' }],
  creator: 'AquaHumanizer',
  publisher: 'AquaHumanizer',
  robots: 'index, follow',
  openGraph: {
    title: 'AquaHumanizer Pro - Professional AI Text Enhancement',
    description: 'Transform AI-generated content with professional styles and ATS optimization',
    type: 'website',
    locale: 'en_US',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'AquaHumanizer Pro',
    description: 'Professional AI text enhancement tool',
  },
}

export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
  themeColor: '#0070F3',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className={inter.variable}>
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
      </head>
      <body className={`${inter.className} antialiased`}>
        {children}
      </body>
    </html>
  )
}
