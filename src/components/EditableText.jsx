import { useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'motion/react'

// A click-to-edit text element used for the poster's headline / subhead /
// date / location. The text content is managed imperatively on the
// contentEditable node so React never clobbers the caret. While editing, a
// subtle highlight sits behind it and an accent underline draws in.
export default function EditableText({
  field,
  value,
  onCommit,
  editingField,
  setEditingField,
  accent,
  style,
  animate,
  transition,
  block = false,
}) {
  const ref = useRef(null)
  const editing = editingField === field

  useEffect(() => {
    const el = ref.current
    if (el && !editing && el.textContent !== value) {
      el.textContent = value
    }
  }, [value, editing])

  return (
    <span
      style={{
        position: 'relative',
        display: block ? 'block' : 'inline-block',
        maxWidth: '100%',
      }}
    >
      <AnimatePresence>
        {editing && (
          <motion.span
            key="highlight"
            className="pointer-events-none"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.18 }}
            style={{
              position: 'absolute',
              inset: '-3px -7px',
              background: 'rgba(255,255,255,0.04)',
              borderRadius: 4,
              zIndex: 0,
            }}
          />
        )}
      </AnimatePresence>

      <motion.div
        ref={ref}
        contentEditable
        suppressContentEditableWarning
        spellCheck={false}
        role="textbox"
        tabIndex={0}
        onClick={() => setEditingField(field)}
        onFocus={() => setEditingField(field)}
        onBlur={(e) => {
          const next = e.currentTarget.textContent.trim() || value
          onCommit(next)
          setEditingField((cur) => (cur === field ? null : cur))
        }}
        onKeyDown={(e) => {
          if (e.key === 'Enter') {
            e.preventDefault()
            e.currentTarget.blur()
          }
        }}
        animate={animate}
        transition={transition}
        style={{
          position: 'relative',
          zIndex: 1,
          outline: 'none',
          cursor: 'text',
          ...style,
        }}
      />

      <AnimatePresence>
        {editing && (
          <motion.span
            key="underline"
            className="pointer-events-none"
            initial={{ scaleX: 0, opacity: 0 }}
            animate={{ scaleX: 1, opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ type: 'spring', stiffness: 400, damping: 30 }}
            style={{
              position: 'absolute',
              left: 0,
              right: 0,
              bottom: -3,
              height: 1,
              background: accent,
              transformOrigin: 'left',
              zIndex: 2,
            }}
          />
        )}
      </AnimatePresence>
    </span>
  )
}
