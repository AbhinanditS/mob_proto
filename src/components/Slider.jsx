import { useRef, useCallback } from 'react'
import { motion } from 'motion/react'

// Custom pointer-driven slider — deliberately NOT an <input type="range">.
// Reused by Typography (size, tracking) and Spacing (margin, gap).
// The fill width animates via Motion so dragging feels physical rather than
// instantaneous.
export default function Slider({ value, min, max, step = 1, accent, onChange }) {
  const trackRef = useRef(null)
  const draggingRef = useRef(false)

  const pct = ((value - min) / (max - min)) * 100

  const setFromClientX = useCallback(
    (clientX) => {
      const el = trackRef.current
      if (!el) return
      const rect = el.getBoundingClientRect()
      const ratio = Math.min(1, Math.max(0, (clientX - rect.left) / rect.width))
      let next = min + ratio * (max - min)
      next = Math.round(next / step) * step
      next = Math.min(max, Math.max(min, next))
      onChange(next)
    },
    [min, max, step, onChange]
  )

  const handlePointerDown = (e) => {
    draggingRef.current = true
    e.currentTarget.setPointerCapture(e.pointerId)
    setFromClientX(e.clientX)
  }

  const handlePointerMove = (e) => {
    if (!draggingRef.current) return
    setFromClientX(e.clientX)
  }

  const handlePointerUp = (e) => {
    draggingRef.current = false
    try {
      e.currentTarget.releasePointerCapture(e.pointerId)
    } catch {
      /* pointer already released */
    }
  }

  return (
    <div
      ref={trackRef}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      onPointerCancel={handlePointerUp}
      className="relative h-4 flex items-center cursor-pointer select-none touch-none"
    >
      {/* Track */}
      <div
        className="relative w-full rounded-full"
        style={{ height: 2, background: 'rgba(255,255,255,0.08)' }}
      >
        {/* Fill */}
        <motion.div
          className="absolute left-0 top-0 h-full rounded-full"
          style={{ background: accent }}
          animate={{ width: `${pct}%` }}
          transition={{ type: 'spring', stiffness: 320, damping: 32 }}
        />
        {/* Thumb */}
        <motion.div
          className="absolute top-1/2"
          style={{ left: `${pct}%` }}
          animate={{ left: `${pct}%` }}
          transition={{ type: 'spring', stiffness: 320, damping: 32 }}
        >
          <div
            className="rounded-full bg-white"
            style={{
              width: 14,
              height: 14,
              transform: 'translate(-50%, -50%)',
              boxShadow: '0 1px 4px rgba(0,0,0,0.5)',
            }}
          />
        </motion.div>
      </div>
    </div>
  )
}
