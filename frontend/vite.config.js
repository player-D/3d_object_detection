import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// https://vite.dev/config/
export default defineConfig({
  plugins: [vue()],
  build: {
    chunkSizeWarningLimit: 700,
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (!id.includes('node_modules')) {
            return
          }
          if (id.includes('three')) {
            return 'vendor-three'
          }
          if (id.includes('element-plus') || id.includes('@element-plus/icons-vue')) {
            return 'vendor-ui'
          }
          if (id.includes('axios')) {
            return 'vendor-http'
          }
          if (id.includes('vue')) {
            return 'vendor-vue'
          }
          return 'vendor'
        },
      },
    },
  },
})
