import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    environment: "node",
    globals: true,
    testTimeout: 1000,
    include: ["test/**/*.test.{js,ts}"],
    exclude: ["node_modules/**/*"],
  },
});
