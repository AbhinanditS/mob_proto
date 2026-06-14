import { motion, AnimatePresence, LayoutGroup } from 'motion/react'
import EditableText from './EditableText'

const POSTER_W = 480
const POSTER_H = 640

const layoutSpring = { type: 'spring', stiffness: 260, damping: 28 }
const typeSpring = { type: 'spring', stiffness: 300, damping: 30 }

// Hidden SVG grain filter — referenced by the poster's grain overlay layer.
function GrainFilter() {
  return (
    <svg style={{ position: 'absolute', width: 0, height: 0 }} aria-hidden>
      <filter id="grain">
        <feTurbulence type="fractalNoise" baseFrequency="0.65" numOctaves="3" stitchTiles="stitch" />
        <feColorMatrix type="saturate" values="0" />
        <feBlend in="SourceGraphic" mode="overlay" result="blend" />
        <feComposite in="blend" in2="SourceGraphic" operator="in" />
      </filter>
    </svg>
  )
}

function Palette({ accent, faint, visible }) {
  return (
    <motion.div
      layout
      className="flex flex-col items-end gap-[6px] shrink-0"
      animate={{ opacity: visible ? 1 : 0 }}
      transition={{ duration: 0.25 }}
      style={{ pointerEvents: visible ? 'auto' : 'none' }}
    >
      <div className="flex gap-[5px]">
        <motion.span
          animate={{ backgroundColor: accent }}
          style={{ width: 10, height: 10, borderRadius: 2 }}
        />
        <span style={{ width: 10, height: 10, borderRadius: 2, background: '#18181A' }} />
      </div>
      <span
        className="font-mono uppercase"
        style={{ fontSize: 7, color: faint, letterSpacing: '0.12em' }}
      >
        Palette
      </span>
    </motion.div>
  )
}

export default function Poster({
  posterSettings,
  posterText,
  setPosterText,
  editingField,
  setEditingField,
}) {
  const {
    layout,
    fontSize,
    fontWeight,
    letterSpacing,
    accentColor,
    darkPoster,
    horizontalMargin,
    verticalGap,
  } = posterSettings

  const isTypo = layout === 2
  const hasImage = layout !== 2
  const isRow = layout === 1
  const imageFirst = layout === 0

  // Ink palette flips with the light/dark poster theme.
  const posterBg = darkPoster ? '#141414' : '#F4F3EF'
  const ink = darkPoster ? '#F0F0F0' : '#111111'
  const sub = darkPoster ? '#9A9A9A' : '#666666'
  const faint = darkPoster ? '#7E7E7E' : '#888888'
  const footerCol = darkPoster ? '#7A7A7A' : '#BBBBBB'
  const imageBg = darkPoster ? '#F4F3EF' : '#18181A'
  const gridLine = darkPoster ? 'rgba(255,255,255,0.04)' : 'rgba(0,0,0,0.045)'

  // Date is stored as "24 OCT" — split for display, recombine on edit.
  const dateNum = posterText.date.split(' ')[0]
  const dateLabel = posterText.date.split(' ').slice(1).join(' ')

  const bodyDirection = isRow ? 'row' : 'column'

  const imageRegion = hasImage && (
    <motion.div
      key="image-region"
      layout
      layoutId="image-region"
      initial={{ opacity: 0, scale: 0.96 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.96 }}
      transition={layoutSpring}
      className="relative overflow-hidden shrink-0"
      style={{
        order: imageFirst ? 0 : 2,
        flexBasis: isRow ? '45%' : '38%',
        width: isRow ? undefined : '100%',
        height: isRow ? '100%' : undefined,
        alignSelf: 'stretch',
      }}
    >
      <motion.div
        className="absolute inset-0"
        animate={{ backgroundColor: imageBg }}
        transition={{ duration: 0.4 }}
      />
      {/* Grain over the image block */}
      <div
        className="absolute inset-0 pointer-events-none"
        style={{ filter: 'url(#grain)', opacity: 0.3, mixBlendMode: 'overlay' }}
      />
      {/* Accent bottom border */}
      <motion.div
        className="absolute left-0 right-0 bottom-0"
        animate={{ backgroundColor: accentColor }}
        style={{ height: 3 }}
      />
    </motion.div>
  )

  return (
    <LayoutGroup>
      <div
        className="relative shrink-0 overflow-hidden"
        style={{
          width: POSTER_W,
          height: POSTER_H,
          boxShadow: '0 8px 48px rgba(0,0,0,0.5), 0 2px 8px rgba(0,0,0,0.3)',
        }}
      >
        <GrainFilter />

        {/* Background color layer (animates on theme toggle) */}
        <motion.div
          className="absolute inset-0"
          animate={{ backgroundColor: posterBg }}
          transition={{ duration: 0.45 }}
        />
        {/* Grid overlay */}
        <div
          className="absolute inset-0 pointer-events-none"
          style={{
            backgroundImage: `linear-gradient(${gridLine} 1px, transparent 1px), linear-gradient(90deg, ${gridLine} 1px, transparent 1px)`,
            backgroundSize: '24px 24px',
          }}
        />

        {/* Content */}
        <div className="relative z-10 flex h-full flex-col">
          {/* Top nav row — always present */}
          <div
            className="flex items-center justify-between shrink-0"
            style={{ padding: '16px 20px 8px' }}
          >
            <span
              className="font-mono uppercase"
              style={{ fontSize: 9, color: faint, letterSpacing: '0.1em' }}
            >
              Creative Story / Event
            </span>
            <span className="font-mono" style={{ fontSize: 9, color: faint }}>
              ←
            </span>
          </div>

          {/* Body — image + text regions arranged per layout */}
          <div
            className="flex flex-1 min-h-0"
            style={{ flexDirection: bodyDirection }}
          >
            <AnimatePresence>{imageRegion}</AnimatePresence>

            {/* Text region */}
            <motion.div
              layout
              layoutId="text-region"
              transition={layoutSpring}
              className="flex flex-col min-h-0 min-w-0"
              style={{ order: 1, flex: '1 1 0%' }}
              animate={{
                paddingLeft: horizontalMargin,
                paddingRight: horizontalMargin,
                paddingTop: verticalGap,
                paddingBottom: verticalGap,
                gap: verticalGap,
              }}
            >
              {/* Header row: kicker + headline + subhead, with palette */}
              <motion.div
                layout
                className="flex min-w-0"
                style={{
                  order: 0,
                  flexGrow: isTypo ? 1 : 0,
                  flexDirection: isTypo ? 'column' : 'row',
                  justifyContent: isTypo ? 'center' : 'space-between',
                  alignItems: isTypo ? 'center' : 'flex-start',
                  gap: isTypo ? verticalGap : 12,
                }}
              >
                <motion.div
                  layout
                  className="flex flex-col min-w-0"
                  style={{
                    gap: 4,
                    alignItems: isTypo ? 'center' : 'flex-start',
                    textAlign: isTypo ? 'center' : 'left',
                  }}
                >
                  <motion.span
                    layout
                    className="font-editorial italic"
                    style={{ fontSize: 13, color: faint }}
                  >
                    the
                  </motion.span>

                  <EditableText
                    field="headline"
                    value={posterText.headline}
                    onCommit={(v) => setPosterText((p) => ({ ...p, headline: v }))}
                    editingField={editingField}
                    setEditingField={setEditingField}
                    accent={accentColor}
                    transition={typeSpring}
                    animate={{
                      fontSize: isTypo ? Math.min(fontSize * 1.7, 132) : fontSize,
                      fontWeight,
                      letterSpacing: `${letterSpacing}px`,
                      color: ink,
                    }}
                    style={{
                      fontFamily: 'Syne, sans-serif',
                      lineHeight: 0.96,
                    }}
                  />

                  <EditableText
                    field="subhead"
                    value={posterText.subhead}
                    onCommit={(v) => setPosterText((p) => ({ ...p, subhead: v }))}
                    editingField={editingField}
                    setEditingField={setEditingField}
                    accent={accentColor}
                    animate={{ color: sub }}
                    style={{
                      fontFamily: '"DM Serif Display", serif',
                      fontStyle: 'italic',
                      fontSize: 11,
                    }}
                  />
                </motion.div>

                <Palette accent={accentColor} faint={faint} visible={!isTypo} />
              </motion.div>

              {/* Spacer pushes the meta cluster to the bottom in non-typo layouts */}
              <div style={{ flexGrow: isTypo ? 0 : 1, order: 1 }} />

              {/* Date cluster */}
              <motion.div
                layout
                className="flex items-baseline gap-2"
                style={{ order: 2 }}
              >
                <EditableText
                  field="date"
                  value={dateNum}
                  onCommit={(v) =>
                    setPosterText((p) => ({
                      ...p,
                      date: `${v}${dateLabel ? ' ' + dateLabel : ''}`,
                    }))
                  }
                  editingField={editingField}
                  setEditingField={setEditingField}
                  accent={accentColor}
                  transition={typeSpring}
                  animate={{ fontSize: isTypo ? 28 : 42, color: ink }}
                  style={{ fontFamily: 'Syne, sans-serif', fontWeight: 700, lineHeight: 1 }}
                />
                {dateLabel && (
                  <span
                    className="font-mono uppercase"
                    style={{ fontSize: 9, color: faint, letterSpacing: '0.06em' }}
                  >
                    {dateLabel}
                  </span>
                )}
              </motion.div>

              {/* Location */}
              <motion.div layout style={{ order: 3 }}>
                <EditableText
                  field="location"
                  value={posterText.location}
                  onCommit={(v) => setPosterText((p) => ({ ...p, location: v }))}
                  editingField={editingField}
                  setEditingField={setEditingField}
                  accent={accentColor}
                  animate={{ color: faint }}
                  style={{
                    fontFamily: '"DM Mono", monospace',
                    fontSize: 8,
                    textTransform: 'uppercase',
                    letterSpacing: '0.08em',
                  }}
                />
              </motion.div>

              {/* Footer — not editable */}
              <motion.div
                layout
                className="font-mono"
                style={{
                  order: 4,
                  fontSize: 8,
                  color: footerCol,
                  letterSpacing: '0.02em',
                }}
              >
                {posterText.footer}
              </motion.div>
            </motion.div>
          </div>
        </div>

        {/* Editing pill floating above the footer */}
        <AnimatePresence>
          {editingField && (
            <motion.div
              key="editing-pill"
              className="absolute left-0 right-0 flex justify-center pointer-events-none"
              style={{ bottom: 40 }}
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 6 }}
              transition={{ duration: 0.2 }}
            >
              <span
                className="font-ui"
                style={{
                  fontSize: 10,
                  color: '#555',
                  background: 'rgba(255,255,255,0.06)',
                  padding: '4px 10px',
                  borderRadius: 999,
                  backdropFilter: 'blur(4px)',
                }}
              >
                Editing — press Enter to confirm
              </span>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </LayoutGroup>
  )
}
