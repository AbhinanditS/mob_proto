import { motion, AnimatePresence } from 'motion/react'
import DefaultControls from './controls/DefaultControls'
import TypographyControls from './controls/TypographyControls'
import ColorControls from './controls/ColorControls'
import SpacingControls from './controls/SpacingControls'
import ThinkingLoader from './ThinkingLoader'
import ChatInput from './ChatInput'

const MODE_LABEL = {
  default: 'Layouts',
  typography: 'Typography',
  color: 'Color',
  spacing: 'Spacing',
}

// Whole-surface entrance: blur + slide in, and cascade its child fields.
const surfaceVariants = {
  initial: { opacity: 0, y: -8, filter: 'blur(6px)' },
  animate: {
    opacity: 1,
    y: 0,
    filter: 'blur(0px)',
    transition: { type: 'spring', stiffness: 320, damping: 28, staggerChildren: 0.055 },
  },
  exit: {
    opacity: 0,
    y: 10,
    filter: 'blur(6px)',
    transition: { duration: 0.16 },
  },
}

function ControlSurface({ controlMode, posterSettings, setPosterSettings }) {
  const props = { posterSettings, setPosterSettings }
  switch (controlMode) {
    case 'typography':
      return <TypographyControls {...props} />
    case 'color':
      return <ColorControls {...props} />
    case 'spacing':
      return <SpacingControls {...props} />
    default:
      return <DefaultControls {...props} />
  }
}

export default function Sidebar({
  controlMode,
  setControlMode,
  posterSettings,
  setPosterSettings,
  inputValue,
  setInputValue,
  isThinking,
  setIsThinking,
}) {
  const accent = posterSettings.accentColor

  return (
    <aside
      className="flex flex-col min-h-0"
      style={{
        flex: '0 0 35%',
        background: '#161616',
        borderLeft: '1px solid rgba(255,255,255,0.06)',
      }}
    >
      {/* Control surface area */}
      <div className="flex-1 min-h-0 overflow-hidden flex flex-col" style={{ padding: 20 }}>
        {/* Section label row */}
        <div className="flex items-center justify-between shrink-0" style={{ height: 20, marginBottom: 22 }}>
          <span
            className="font-ui uppercase"
            style={{ fontSize: 11, color: '#555', letterSpacing: '0.08em' }}
          >
            {MODE_LABEL[isThinking ? 'default' : controlMode]}
          </span>
          <AnimatePresence>
            {controlMode !== 'default' && !isThinking && (
              <motion.button
                key="back"
                type="button"
                onClick={() => setControlMode('default')}
                className="font-ui"
                style={{ fontSize: 11, color: '#555', background: 'none', border: 'none', cursor: 'pointer' }}
                initial={{ opacity: 0, x: 8 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 8 }}
                whileHover={{ color: '#999' }}
                transition={{ duration: 0.2 }}
              >
                ← Back
              </motion.button>
            )}
          </AnimatePresence>
        </div>

        {/* Swappable surface / thinking loader */}
        <div className="flex-1 min-h-0 overflow-y-auto">
          <AnimatePresence mode="wait">
            {isThinking ? (
              <ThinkingLoader key="thinking" accent={accent} />
            ) : (
              <motion.div
                key={controlMode}
                variants={surfaceVariants}
                initial="initial"
                animate="animate"
                exit="exit"
              >
                <ControlSurface
                  controlMode={controlMode}
                  posterSettings={posterSettings}
                  setPosterSettings={setPosterSettings}
                />
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>

      {/* Hairline divider */}
      <div style={{ height: 1, background: 'rgba(255,255,255,0.06)' }} />

      {/* Chat input pinned at bottom */}
      <ChatInput
        inputValue={inputValue}
        setInputValue={setInputValue}
        accent={accent}
        isThinking={isThinking}
        setIsThinking={setIsThinking}
        setControlMode={setControlMode}
      />
    </aside>
  )
}
