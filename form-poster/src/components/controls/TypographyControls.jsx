import Slider from '../Slider'
import Pill from '../Pill'
import { Field, LabelValueRow, ControlLabel } from './shared'

const WEIGHTS = [
  { label: 'Thin', value: 300 },
  { label: 'Regular', value: 400 },
  { label: 'Bold', value: 700 },
  { label: 'Black', value: 800 },
]

export default function TypographyControls({ posterSettings, setPosterSettings }) {
  const { fontSize, fontWeight, letterSpacing, accentColor } = posterSettings
  const set = (patch) => setPosterSettings((p) => ({ ...p, ...patch }))

  const trackingLabel = `${letterSpacing < 0 ? '−' : ''}${Math.abs(letterSpacing)}px`

  return (
    <div className="flex flex-col" style={{ gap: 26 }}>
      <Field>
        <LabelValueRow label="Size" value={`${fontSize}px`} accent={accentColor} />
        <Slider value={fontSize} min={32} max={96} accent={accentColor} onChange={(v) => set({ fontSize: v })} />
      </Field>

      <Field>
        <div style={{ marginBottom: 10 }}>
          <ControlLabel>Weight</ControlLabel>
        </div>
        <div className="flex" style={{ gap: 6 }}>
          {WEIGHTS.map((w) => (
            <Pill
              key={w.value}
              label={w.label}
              selected={fontWeight === w.value}
              accent={accentColor}
              onClick={() => set({ fontWeight: w.value })}
            />
          ))}
        </div>
      </Field>

      <Field>
        <LabelValueRow label="Tracking" value={trackingLabel} accent={accentColor} />
        <Slider value={letterSpacing} min={-4} max={4} accent={accentColor} onChange={(v) => set({ letterSpacing: v })} />
      </Field>
    </div>
  )
}
