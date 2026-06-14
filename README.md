# FORM — Poster Canvas

An interactive demo of a desktop poster-making tool. Compose an editorial poster
and reshape it live: switch between four layouts, tune typography, recolor, and
adjust spacing — all driven by physical spring animations. A chat input routes
plain-language requests ("change the font", "recolor it") to the matching control
surface with a brief generating moment.

Built to be screen-recordable and deploy-ready.

## Stack

- **Vite + React** — single full-viewport screen, no scrolling
- **[Motion](https://motion.dev)** (`motion/react`) — every transition is a Motion
  animation; there are no CSS transitions anywhere
- **Tailwind CSS v3** — utility styling, custom design tokens
- No component libraries — everything is hand-built

## Develop

```bash
npm install
npm run dev      # http://localhost:5173
npm run build    # production build → dist/
npm run lint
```

## Highlights

- **Four morphing layouts** (Editorial / Split / Typographic / Inverted) — shared
  elements animate to new positions via Motion layout animations.
- **Live typography** — size / weight / tracking sliders animate the poster text
  in place.
- **Accent propagation** — one accent color drives both the poster and the tool's
  own chrome (send button, selected states), recoloring everything at once.
- **Click-to-edit** — headline, subhead, date, and location are editable in place.
- **Chat-driven controls** — keyword intent detection swaps control surfaces with a
  blur-in "generating" moment.

## Deploy (Vercel)

`vercel.json` and `.nvmrc` are included. Point Vercel at this directory; it builds
with `npm run build` and serves `dist/`.
