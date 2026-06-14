import { motion } from 'motion/react'

// Reusable selectable pill — used by Typography (weight) and Color (theme).
// Selected state animates background + text color via Motion (no CSS transition).
export default function Pill({ label, selected, accent, onClick }) {
  return (
    <motion.button
      type="button"
      onClick={onClick}
      whileTap={{ scale: 0.95 }}
      transition={{ type: 'spring', stiffness: 400, damping: 20 }}
      className="flex-1 rounded-[7px] font-ui"
      style={{ height: 28, fontSize: 11, border: 'none', cursor: 'pointer' }}
      initial={false}
    >
      <motion.span
        className="flex h-full w-full items-center justify-center rounded-[7px]"
        animate={{
          backgroundColor: selected ? accent : 'rgba(255,255,255,0.04)',
          color: selected ? '#ffffff' : '#666666',
        }}
        transition={{ duration: 0.22, ease: 'easeOut' }}
      >
        {label}
      </motion.span>
    </motion.button>
  )
}
