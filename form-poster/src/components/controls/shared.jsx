import { motion } from 'motion/react'
import { itemVariants } from './variants'

// A single staggered field block within a control surface.
export function Field({ children, className = '', style }) {
  return (
    <motion.div variants={itemVariants} className={className} style={style}>
      {children}
    </motion.div>
  )
}

// Small uppercase control label, e.g. "SIZE".
export function ControlLabel({ children }) {
  return (
    <span
      className="font-mono uppercase"
      style={{ fontSize: 10, color: '#555', letterSpacing: '0.04em' }}
    >
      {children}
    </span>
  )
}

// Label on the left, live value (in accent) on the right.
export function LabelValueRow({ label, value, accent }) {
  return (
    <div className="flex items-center justify-between" style={{ marginBottom: 10 }}>
      <ControlLabel>{label}</ControlLabel>
      <span
        className="font-mono"
        style={{ fontSize: 10, color: accent, letterSpacing: '0.02em' }}
      >
        {value}
      </span>
    </div>
  )
}
