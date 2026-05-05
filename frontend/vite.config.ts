import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from 'path'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (!id.includes('node_modules')) return
          if (
            id.includes('/react/') ||
            id.includes('/react-dom/') ||
            id.includes('/react-router/') ||
            id.includes('/react-router-dom/') ||
            id.includes('/scheduler/') ||
            id.includes('/use-sync-external-store/')
          ) {
            return 'vendor-react'
          }
          if (id.includes('/@tanstack/')) return 'vendor-query'
          if (id.includes('/axios/')) return 'vendor-http'
          if (id.includes('/lucide-react/')) return 'vendor-icons'
          if (
            id.includes('/recharts/') ||
            id.includes('/d3-') ||
            id.includes('/victory-vendor/')
          ) {
            return 'vendor-charts'
          }
          if (
            id.includes('/react-markdown/') ||
            id.includes('/remark-') ||
            id.includes('/unified/') ||
            id.includes('/hast-') ||
            id.includes('/mdast-') ||
            id.includes('/micromark') ||
            id.includes('/unist-') ||
            id.includes('/vfile') ||
            id.includes('/property-information/') ||
            id.includes('/space-separated-tokens/') ||
            id.includes('/comma-separated-tokens/') ||
            id.includes('/character-entities') ||
            id.includes('/decode-named-character-reference/') ||
            id.includes('/trim-lines/') ||
            id.includes('/zwitch/')
          ) {
            return 'vendor-markdown'
          }
          return 'vendor'
        },
      },
    },
  },
})
