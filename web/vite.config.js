import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// For GitHub Pages PROJECT sites the app is served from a subpath
// (https://<user>.github.io/<repo>/), so the asset base must be that subpath.
// Override at build time:  VITE_BASE=/rmc-phonon-dynamics/ npm run build
// Defaults to '/' for local dev and user/organization Pages.
export default defineConfig({
  base: process.env.VITE_BASE || '/',
  plugins: [
    react(),
    tailwindcss(),
  ],
  build: {
    // Three.js core (~490 kB minified) is isolated into its own lazy vendor chunk
    // and only loaded when a 3D crystal / Brillouin-zone / domain canvas mounts.
    chunkSizeWarningLimit: 600,
    rolldownOptions: {
      output: {
        codeSplitting: {
          groups: [
            {
              name: 'vendor-three',
              test: /node_modules[\\/]three/,
              priority: 20,
            },
            {
              name: 'vendor-react',
              test: /node_modules[\\/](react|react-dom|scheduler)/,
              priority: 15,
            },
          ],
        },
      },
    },
  },
})
