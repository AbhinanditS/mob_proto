// Per-item entrance used by every control field. The staggered timing comes
// from the parent surface's staggerChildren; this just declares the transform.
export const itemVariants = {
  initial: { opacity: 0, y: -6, filter: 'blur(4px)' },
  animate: {
    opacity: 1,
    y: 0,
    filter: 'blur(0px)',
    transition: { type: 'spring', stiffness: 320, damping: 26 },
  },
}
