<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import Router, { link } from "svelte-spa-router";
  import active from "svelte-spa-router/active";
  import wrap from "svelte-spa-router/wrap";

  import Home from "./Home.svelte";
  import ReactWrapper from "./react/ReactWrapper.svelte";
  import EmbeddingAtlasExample from "./svelte/EmbeddingAtlasExample.svelte";
  import EmbeddingViewExample from "./svelte/EmbeddingViewExample.svelte";
  import EmbeddingViewMosaicExample from "./svelte/EmbeddingViewMosaicExample.svelte";
  import FindClustersExample from "./svelte/FindClustersExample.svelte";

  import ReactEmbeddingAtlasExample from "./react/EmbeddingAtlas.js";
  import ReactEmbeddingViewExample from "./react/EmbeddingView.js";

  function reactWrap(component: any) {
    return wrap({ component: ReactWrapper as any, props: { component: component } });
  }

  const routes = {
    "/": Home,
    "/svelte/embedding-atlas": EmbeddingAtlasExample,
    "/svelte/embedding-view": EmbeddingViewExample,
    "/svelte/embedding-view-mosaic": EmbeddingViewMosaicExample,
    "/svelte/find-clusters": FindClustersExample,
    "/react/embedding-view": reactWrap(ReactEmbeddingViewExample),
    "/react/embedding-atlas": reactWrap(ReactEmbeddingAtlasExample),
  };

  const links = [
    {
      title: "Svelte Examples",
      items: [
        { title: "EmbeddingView", href: "/svelte/embedding-view" },
        { title: "EmbeddingViewMosaic", href: "/svelte/embedding-view-mosaic" },
        { title: "EmbeddingAtlas", href: "/svelte/embedding-atlas" },
        { title: "findClusters", href: "/svelte/find-clusters" },
      ],
    },
    {
      title: "React Examples",
      items: [
        { title: "EmbeddingView", href: "/react/embedding-view" },
        { title: "EmbeddingAtlas", href: "/react/embedding-atlas" },
      ],
    },
  ];
</script>

<div class="fixed left-0 top-0 right-0 bottom-0 flex">
  <div class="w-96 flex-none bg-slate-100 p-2 border-r border-slate-200">
    {#each links as { title, items }}
      <h2 class="font-bold mb-1">{title}</h2>
      {#each items as { title, href }}
        <a
          class="block px-2 py-1 bg-slate-300 mb-1 rounded-sm hover:bg-slate-400 hover:text-slate-100"
          href={href}
          use:link
          use:active={{ className: "bg-slate-500! text-slate-100!" }}
        >
          <code>{title}</code>
        </a>
      {/each}
      <div class="mb-4"></div>
    {/each}
  </div>
  <div class="p-2 flex-1 overflow-x-hidden overflow-y-scroll">
    <Router routes={routes} />
  </div>
</div>
