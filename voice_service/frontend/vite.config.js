import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The FastAPI voice service runs on :8001. We proxy the API paths through the
// Vite dev server so the browser talks to a single origin (no CORS), and the
// mic works because the page is served from http://localhost (secure context).
const API = "http://localhost:8001";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/ask_voice": API,
      "/health": API,
      "/stt": API,
      "/tts": API,
    },
  },
});
