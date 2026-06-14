import { motion } from 'motion/react'

// The hero moment: three accent dots bouncing while the tool "generates"
// a new control surface, with a label that fades in slightly after.
export default function ThinkingLoader({ accent }) {
  const dots = [0, 0.14, 0.28]

  return (
    <motion.div
      className="flex flex-col items-center justify-center gap-4 py-10"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.18 }}
    >
      <div className="flex items-end gap-[6px]" style={{ height: 12 }}>
        {dots.map((delay, i) => (
          <motion.div
            key={i}
            style={{ width: 5, height: 5, borderRadius: '50%', background: accent }}
            animate={{ y: [0, -7, 0] }}
            transition={{
              duration: 0.62,
              repeat: Infinity,
              ease: 'easeInOut',
              delay,
            }}
          />
        ))}
      </div>
      <motion.div
        className="font-mono"
        style={{ fontSize: 10, color: '#444', letterSpacing: '0.02em' }}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.2, duration: 0.3 }}
      >
        Generating controls...
      </motion.div>
    </motion.div>
  )
}
