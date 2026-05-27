import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  server: {
    proxy: {
      '/live': {
        target: 'http://192.168.151.158:8081',
        changeOrigin: true,
        // ws: true, // 如果是 WebSocket 需要开启
      }
    },
    port: 4173
  }
})