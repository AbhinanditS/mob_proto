import { motion } from 'motion/react'
import { ShareIcon, ExportIcon, MoreIcon } from './icons'

function GhostButton({ children, label }) {
  return (
    <motion.button
      type="button"
      aria-label={label}
      title={label}
      className="grid place-items-center"
      style={{ width: 32, height: 32, borderRadius: 8, color: '#555', border: 'none', background: 'transparent', cursor: 'pointer' }}
      initial={false}
      whileHover={{ backgroundColor: 'rgba(255,255,255,0.05)', color: '#999' }}
      whileTap={{ scale: 0.95 }}
      transition={{ type: 'spring', stiffness: 400, damping: 20 }}
    >
      {children}
    </motion.button>
  )
}

export default function Topbar() {
  return (
    <header
      className="flex items-center justify-between shrink-0"
      style={{
        height: 48,
        background: '#111111',
        borderBottom: '1px solid rgba(255,255,255,0.06)',
        paddingLeft: 18,
        paddingRight: 12,
      }}
    >
      <div className="flex items-center" style={{ gap: 10 }}>
        <span className="font-display" style={{ fontWeight: 500, fontSize: 14, color: '#F0F0F0', letterSpacing: '0.01em' }}>
          FORM
        </span>
        <span style={{ color: '#333', fontSize: 13 }}>/</span>
        <span className="font-ui" style={{ fontSize: 13, color: '#555' }}>
          Poster Canvas
        </span>
      </div>

      <div className="flex items-center" style={{ gap: 2 }}>
        <GhostButton label="Share"><ShareIcon /></GhostButton>
        <GhostButton label="Export"><ExportIcon /></GhostButton>
        <GhostButton label="More"><MoreIcon /></GhostButton>
      </div>
    </header>
  )
}
