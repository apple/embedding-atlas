import { defineConfig } from "vitepress";

import docComment from "./doc_comment.mjs";

// https://vitepress.dev/reference/site-config
export default defineConfig({
  base: "/embedding-atlas/",
  title: "Embedding Atlas",
  description:
    "Interactive visualizations for large-scale embeddings. Effortlessly explore, filter, and search through rich metadata.",
  head: [
    ["link", { rel: "icon", href: "/embedding-atlas/favicon.svg", media: "(prefers-color-scheme: light)" }],
    ["link", { rel: "icon", href: "/embedding-atlas/favicon_dark.svg", media: "(prefers-color-scheme: dark)" }],
    ["meta", { property: "twitter:image", content: "/embedding-atlas/social.png" }],
    ["meta", { property: "og:image", content: "/embedding-atlas/social.png" }],
  ],
  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    logo: {
      light: "/favicon.svg",
      dark: "/favicon_dark.svg",
    },
    search: {
      provider: "local",
    },
    nav: [
      { text: "Home", link: "/" },
      { text: "Docs", link: "/overview", target: "_self" },
      { text: "Demo", link: "/demo/index.html", target: "_self" },
    ],
    sidebar: [
      { text: "Overview", link: "/overview" },
      {
        text: "Use Embedding Atlas",
        items: [
          { text: "Command Line Utility", link: "/tool" },
          { text: "Python Notebook Widget", link: "/widget" },
          { text: "Streamlit Component", link: "/streamlit" },
        ],
      },
      {
        text: "Component Library",
        items: [
          { text: "Table", link: "/table" },
          { text: "EmbeddingView", link: "/embedding-view" },
          { text: "EmbeddingViewMosaic", link: "/embedding-view-mosaic" },
          { text: "EmbeddingAtlas", link: "/embedding-atlas" },
          { text: "Algorithms", link: "/algorithms" },
        ],
      },
      {
        text: "Development",
        items: [{ text: "Development Instructions", link: "/develop" }],
      },
    ],
    socialLinks: [{ icon: "github", link: "https://github.com/apple/embedding-atlas" }],
    footer: {
      copyright: "Copyright © 2025 Apple Inc. Released under the MIT license.",
    },
  },
  vite: {
    clearScreen: false,
  },
  markdown: {
    config: (md) => {
      md.use(docComment, { indexDts: "../embedding-atlas/dist/index.d.ts" });
    },
  },
});
