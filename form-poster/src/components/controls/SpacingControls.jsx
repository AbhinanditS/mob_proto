import { motion } from 'motion/react'
import Slider from '../Slider'
import { Field, LabelValueRow } from './shared'

// Live preview: outer rect = poster, inner rect = content area. The margin gap
// and internal line rhythm track the two sliders.
function SpacingDiagram({ horizontalMargin, verticalGap }) {
  const W = 120
  const H = 80
  // Map margin (12–40) to an inset in diagram units.
  const insetX = 6 + ((horizontalMargin - 12) / (40 - 12)) * 16
  const insetY = 8 + ((horizontalMargin - 12) / (40 - 12)) * 8
  const innerW = W - insetX * 2
  const innerH = H - insetY * 2

  // Map gap (8–32) to line spacing inside the content area.
  const spacing = 7 + ((verticalGap - 8) / (32 - 8)) * 11
  const lineTop = insetY + 8
  const lines = [0, 1, 2].map((i) => lineTop + i * spacing)
  const tr = { type: 'spring', stiffness: 260, damping: 30 }

  return (
    <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`}>
      <rect x="0.5" y="0.5" width={W - 1} height={H - 1} rx="4" fill="none" stroke="rgba(255,255,255,0.1)" />
      <motion.rect
        rx="3"
        fill="rgba(255,255,255,0.05)"
        stroke="rgba(255,255,255,0.15)"
        animate={{ x: insetX, y: insetY, width: innerW, height: innerH }}
        transition={tr}
      />
      {lines.map((y, i) => (
        <motion.rect
          key={i}
          height="2.5"
          rx="1.25"
          fill="rgba(255,255,255,0.22)"
          animate={{ x: insetX + 8, y, width: innerW - 16 - (i === 2 ? 14 : 0) }}
          transition={tr}
        />
      ))}
    </svg>
  )
}

export default function SpacingControls({ posterSettings, setPosterSettings }) {
  const { horizontalMargin, verticalGap, accentColor } = posterSettings
  const set = (patch) => setPosterSettings((p) => ({ ...p, ...patch }))

  return (
    <div className="flex flex-col" style={{ gap: 26 }}>
      <Field>
        <LabelValueRow label="Horizontal Margin" value={`${horizontalMargin}px`} accent={accentColor} />
        <Slider value={horizontalMargin} min={12} max={40} accent={accentColor} onChange={(v) => set({ horizontalMargin: v })} />
      </Field>

      <Field>
        <LabelValueRow label="Vertical Gap" value={`${verticalGap}px`} accent={accentColor} />
        <Slider value={verticalGap} min={8} max={32} accent={accentColor} onChange={(v) => set({ verticalGap: v })} />
      </Field>

      <Field className="flex justify-center" style={{ paddingTop: 8 }}>
        <SpacingDiagram horizontalMargin={horizontalMargin} verticalGap={verticalGap} />
      </Field>
    </div>
  )
}
