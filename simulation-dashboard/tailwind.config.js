/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx,ts,tsx}'],
  theme: {
    extend: {
      colors: {
        'simulation-blue': '#1e40af',
        'simulation-green': '#059669',
        'simulation-purple': '#7c3aed',
        'simulation-orange': '#ea580c',
      },
    },
  },
  plugins: [],
}