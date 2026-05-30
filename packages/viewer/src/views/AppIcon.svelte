<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import iconDark from "../assets/icon-dark.png?url";
  import iconLight from "../assets/icon-light.png?url";

  interface Props {
    colorScheme?: "light" | "dark";
  }

  let { colorScheme = "light" }: Props = $props();

  const iconSize = 33;
</script>

<a
  class="w-[28px] h-[28px] flex items-center justify-center cursor-pointer"
  href="https://apple.github.io/embedding-atlas/"
  target="_blank"
  rel="noopener noreferrer"
  title="Embedding Atlas"
>
  <div class="shrink-0" style:width="{iconSize}px" style:height="{iconSize}px">
    {#if colorScheme == "light"}
      <img src={iconLight} alt="Embedding Atlas Icon" class="w-full" />
    {:else}
      <img src={iconDark} alt="Embedding Atlas Icon" class="w-full" />
    {/if}
  </div>
</a>
