import { motion } from 'motion/react'
import { Field } from './shared'

// Abstract per-layout diagrams. White rects = image block, faint lines = text.
function LayoutDiagram({ index }) {
  const line = (x, y, w, h = 3, op = 0.22) => (
    <rect x={x} y={y} width={w} height={h} rx={1.5} fill="#fff" opacity={op} />
  )
  const block = (x, y, w, h) => (
    <rect x={x} y={y} width={w} height={h} rx={2} fill="#fff" opacity={0.16} />
  )

  return (
    <svg width="80" height="68" viewBox="0 0 80 68">
      {index === 0 && (
        <>
          {block(8, 6, 64, 28)}
          {line(8, 42, 44)}
          {line(8, 50, 32)}
          {line(8, 58, 38)}
        </>
      )}
      {index === 1 && (
        <>
          {block(44, 6, 28, 56)}
          {line(8, 14, 28)}
          {line(8, 24, 22)}
          {line(8, 40, 26)}
          {line(8, 48, 18)}
        </>
      )}
      {index === 2 && (
        <>
          {line(10, 22, 60, 7, 0.26)}
          {line(10, 34, 44, 7, 0.26)}
          {line(10, 56, 24, 2.5, 0.2)}
          {line(10, 61, 18, 2.5, 0.2)}
        </>
      )}
      {index === 3 && (
        <>
          {line(8, 8, 44)}
          {line(8, 16, 32)}
          {line(8, 24, 38)}
          {block(8, 34, 64, 28)}
        </>
      )}
    </svg>
  )
}

const LAYOUTS = [
  { i: 0, label: 'Editorial' },
  { i: 1, label: 'Split' },
  { i: 2, label: 'Typo' },
  { i: 3, label: 'Invert' },
]

export default function DefaultControls({ posterSettings, setPosterSettings }) {
  const { layout, accentColor } = posterSettings

  return (
    <Field>
      <div className="grid grid-cols-2" style={{ gap: 12, justifyItems: 'start' }}>
        {LAYOUTS.map(({ i, label }) => {
          const selected = layout === i
          return (
            <div key={i} className="flex flex-col" style={{ gap: 6 }}>
              <motion.button
                type="button"
                onClick={() =>
                  setPosterSettings((p) => ({ ...p, layout: i }))
                }
                whileTap={{ scale: 0.95 }}
                transition={{ type: 'spring', stiffness: 400, damping: 20 }}
                className="grid place-items-center"
                style={{
                  width: 88,
                  height: 88,
                  borderRadius: 10,
                  background: '#1E1E1E',
                  border: '1px solid',
                  padding: 10,
                  cursor: 'pointer',
                }}
                initial={false}
                animate={{
                  borderColor: selected ? accentColor : 'rgba(255,255,255,0.07)',
                  boxShadow: selected
                    ? `0 0 0 1px ${accentColor}`
                    : '0 0 0 0px rgba(0,0,0,0)',
                }}
              >
                <LayoutDiagram index={i} />
              </motion.button>
              <span
                className="font-mono"
                style={{ fontSize: 9, color: selected ? '#888' : '#444' }}
              >
                {label}
              </span>
            </div>
          )
        })}
      </div>
    </Field>
  )
}
