<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { audioToDataUrl } from "@embedding-atlas/utils";

  import AudioPlayer from "./AudioPlayer.svelte";

  interface Props {
    value: any;
  }

  let { value }: Props = $props();

  let src = $derived(audioToDataUrl(value));
</script>

<AudioPlayer src={src} />
