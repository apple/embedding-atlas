<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import AudioPlayer from "./AudioPlayer.svelte";

  import { audioToDataUrl } from "../utils/media.js";

  interface Props {
    value: any;
  }

  let { value }: Props = $props();

  let src = $derived(audioToDataUrl(value));
</script>

<AudioPlayer src={src} />
