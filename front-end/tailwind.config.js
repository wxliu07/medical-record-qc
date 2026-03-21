/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        bg: '#f8fafc',
        card: '#ffffff',
        border: '#e2e8f0',
        primary: '#3b82f6',
        success: '#22c55e',
        warning: '#f59e0b',
        error: '#ef4444',
        text: '#1e293b',
        'text-muted': '#64748b',
      },
    },
  },
  plugins: [],
}
