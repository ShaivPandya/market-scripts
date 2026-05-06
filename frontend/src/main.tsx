import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import { reloadOnceForStaleAssetLoad } from '@/lib/chunkRecovery'

type VitePreloadErrorEvent = Event & { payload?: unknown }

window.addEventListener('vite:preloadError', (event: VitePreloadErrorEvent) => {
  event.preventDefault()
  reloadOnceForStaleAssetLoad('vite-preload', event.payload)
})

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
