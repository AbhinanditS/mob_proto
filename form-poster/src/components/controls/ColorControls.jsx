import { motion } from 'motion/react'
import Pill from '../Pill'
import { Field, ControlLabel } from './shared'

const ACCENTS = ['#B85C3A', '#7C8B40', '#4A6FA5', '#C4A44B', '#8B3A52', '#3A6B5A']

export default function ColorControls({ posterSettings, setPosterSettings }) {
  const { accentColor, darkPoster } = posterSettings
  const set = (patch) => setPosterSettings((p) => ({ ...p, ...patch }))

  return (
    <div className="flex flex-col" style={{ gap: 26 }}>
      <Field>
        <div style={{ marginBottom: 12 }}>
          <ControlLabel>Accent</ControlLabel>
        </div>
        <div className="flex" style={{ gap: 10 }}>
          {ACCENTS.map((c) => {
            const selected = accentColor.toLowerCase() === c.toLowerCase()
            return (
              <motion.button
                key={c}
                type="button"
                onClick={() => set({ accentColor: c })}
                whileTap={{ scale: 0.9 }}
                transition={{ type: 'spring', stiffness: 400, damping: 20 }}
                style={{
                  width: 28,
                  height: 28,
                  borderRadius: '50%',
                  background: c,
                  border: 'none',
                  cursor: 'pointer',
                  outlineOffset: 2,
                }}
                initial={false}
                animate={{
                  outline: selected
                    ? '2px solid #ffffff'
                    : '2px solid rgba(255,255,255,0)',
                }}
              />
            )
          })}
        </div>
      </Field>

      <Field>
        <div style={{ marginBottom: 10 }}>
          <ControlLabel>Theme</ControlLabel>
        </div>
        <div className="flex" style={{ gap: 6 }}>
          <Pill
            label="Light"
            selected={!darkPoster}
            accent={accentColor}
            onClick={() => set({ darkPoster: false })}
          />
          <Pill
            label="Dark"
            selected={darkPoster}
            accent={accentColor}
            onClick={() => set({ darkPoster: true })}
          />
        </div>
      </Field>
    </div>
  )
}
