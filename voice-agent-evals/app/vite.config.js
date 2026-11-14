import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Build output goes to frontend/dist, which FastAPI serves at "/".
// For local development, `npm run dev` proxies /api to a uvicorn server on port 8000.
export default defineConfig({
  root: "frontend",
  plugins: [react()],
  build: { outDir: "dist", emptyOutDir: true },
  server: { proxy: { "/api": "http://localhost:8000" } },
});
