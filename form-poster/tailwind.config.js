/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      fontFamily: {
        display: ['Syne', 'sans-serif'],
        mono: ['"DM Mono"', 'monospace'],
        editorial: ['"DM Serif Display"', 'serif'],
        ui: ['system-ui', '-apple-system', 'BlinkMacSystemFont', 'sans-serif'],
      },
      colors: {
        app: '#0E0E0E',
        topbar: '#111111',
        sidebar: '#161616',
        surface: '#1E1E1E',
      },
    },
  },
  plugins: [],
}
