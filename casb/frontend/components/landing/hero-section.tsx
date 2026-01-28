'use client'

import { motion } from 'framer-motion'
import Image from 'next/image'

function BlobBackground() {
  return (
    <div className="pointer-events-none absolute inset-0 opacity-60">
      <svg width="100%" height="100%" className="absolute inset-0">
        <defs>
          <linearGradient id="hero-gradient" x1="0" x2="1" y1="0" y2="1">
            <stop offset="0%" stopColor="rgba(59, 130, 246, 0.15)" />
            <stop offset="100%" stopColor="rgba(56, 189, 248, 0.15)" />
          </linearGradient>
          <filter id="hero-filter">
            <feGaussianBlur in="SourceGraphic" stdDeviation="6" result="blur" />
            <feColorMatrix in="blur" mode="matrix" values="1 0 0 0 0  0 1 0 0 0  0 0 1 0 0  0 0 0 20 -10" result="blob" />
          </filter>
        </defs>
        <circle cx="50%" cy="50%" r="40%" fill="url(#hero-gradient)" filter="url(#hero-filter)" />
      </svg>
    </div>
  )
}

export function HeroSection() {
  return (
    <div className="text-center mb-4 relative pt-12">
      <div className="absolute inset-0 overflow-hidden -z-10">
        <BlobBackground />
      </div>
      
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.7, ease: "easeOut" }}
        className="relative"
      >
        <motion.h1
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.7, delay: 0.2 }}
          className="text-4xl md:text-5xl lg:text-6xl font-bold mb-6 bg-clip-text"
        >
          Cloud Services API Security
        </motion.h1>
      </motion.div>
    </div>
  )
} 