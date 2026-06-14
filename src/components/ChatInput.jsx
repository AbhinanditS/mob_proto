import { useState } from 'react'
import { motion, useAnimationControls } from 'motion/react'
import { ArrowUpIcon } from './icons'

const INTENT_KEYWORDS = {
  typography: ['typo', 'font', 'text', 'letter', 'size', 'weight', 'type', 'heading', 'tracking'],
  color: ['color', 'colour', 'palette', 'hue', 'tone', 'accent', 'theme'],
  spacing: ['space', 'spacing', 'margin', 'padding', 'gap', 'rhythm'],
}

function detectIntent(text) {
  const t = text.toLowerCase()
  for (const [intent, words] of Object.entries(INTENT_KEYWORDS)) {
    if (words.some((w) => t.includes(w))) return intent
  }
  return 'default'
}

export default function ChatInput({
  inputValue,
  setInputValue,
  accent,
  isThinking,
  setIsThinking,
  setControlMode,
}) {
  const [focused, setFocused] = useState(false)
  const glow = useAnimationControls()

  const submit = () => {
    const text = inputValue.trim()
    if (!text || isThinking) return

    // Brief glow pulse, then "thinking", then reveal the matched control surface.
    glow.start({
      boxShadow: [
        `0 0 0 2px ${accent}55, 0 0 24px ${accent}33`,
        '0 0 0 0px rgba(255,255,255,0), 0 0 0px rgba(255,255,255,0)',
      ],
      transition: { duration: 0.4, ease: 'easeOut' },
    })

    const intent = detectIntent(text)
    setIsThinking(true)
    setInputValue('')

    setTimeout(() => {
      setIsThinking(false)
      setControlMode(intent)
    }, 750)
  }

  const focusShadow = focused
    ? '0 0 0 2px rgba(255,255,255,0.05), 0 0 20px rgba(255,255,255,0.03)'
    : '0 0 0 0px rgba(255,255,255,0), 0 0 0px rgba(255,255,255,0)'

  return (
    <div style={{ padding: 16 }}>
      <motion.div
        className="relative flex items-center"
        style={{
          height: 44,
          background: '#1E1E1E',
          borderRadius: 12,
          border: '1px solid',
          padding: '0 8px 0 12px',
        }}
        initial={false}
        animate={{
          boxShadow: focusShadow,
          borderColor: focused ? 'rgba(255,255,255,0.14)' : 'rgba(255,255,255,0.07)',
        }}
        transition={{ duration: 0.25 }}
      >
        {/* Glow pulse layer driven imperatively on submit */}
        <motion.div
          className="absolute inset-0 pointer-events-none"
          style={{ borderRadius: 12 }}
          animate={glow}
        />

        <input
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onFocus={() => setFocused(true)}
          onBlur={() => setFocused(false)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') {
              e.preventDefault()
              submit()
            }
          }}
          placeholder="What would you like to edit?"
          className="flex-1 bg-transparent font-ui"
          style={{ fontSize: 13, color: '#F0F0F0', border: 'none', outline: 'none' }}
        />

        <motion.button
          type="button"
          aria-label="Send"
          onClick={submit}
          whileTap={{ scale: 0.92 }}
          transition={{ type: 'spring', stiffness: 400, damping: 20 }}
          className="grid place-items-center shrink-0"
          style={{ width: 28, height: 28, borderRadius: '50%', border: 'none', cursor: 'pointer', color: '#fff' }}
          initial={false}
          animate={{ backgroundColor: accent }}
        >
          <ArrowUpIcon />
        </motion.button>
      </motion.div>
    </div>
  )
}
