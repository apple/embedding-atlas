<script lang="ts" module>
  let currentlyPlaying: HTMLAudioElement | null = null;
</script>

<script lang="ts">
  import { interactionHandler, type CursorValue } from "@embedding-atlas/utils";
  import { IconPause, IconPlay } from "../assets/icons.js";

  interface Props {
    src?: string | null;
  }

  let { src }: Props = $props();

  let audio: HTMLAudioElement | undefined = $state();
  let progressBar: HTMLDivElement | undefined = $state();
  let paused = $state(true);
  let currentTime = $state(0);
  let duration = $state(0);
  let dragging = $state(false);

  function formatTime(seconds: number): string {
    if (!isFinite(seconds) || seconds < 0) return "0:00";
    let m = Math.floor(seconds / 60);
    let s = Math.floor(seconds % 60);
    return `${m}:${s.toString().padStart(2, "0")}`;
  }

  function togglePlay() {
    if (!audio) return;
    if (paused) {
      if (currentlyPlaying && currentlyPlaying !== audio) {
        currentlyPlaying.pause();
        currentlyPlaying.currentTime = 0;
      }
      currentlyPlaying = audio;
      audio.play();
    } else {
      audio.pause();
    }
  }

  function seekFromCursor(cursor: CursorValue) {
    if (!progressBar || !audio || !isFinite(duration) || duration <= 0) return;
    const rect = progressBar.getBoundingClientRect();
    const ratio = Math.max(0, Math.min(1, (cursor.clientX - rect.left) / rect.width));
    audio.currentTime = ratio * duration;
  }

  function startDrag(e: CursorValue) {
    dragging = true;
    seekFromCursor(e);
    return {
      move: (e: CursorValue) => seekFromCursor(e),
      up: () => {
        dragging = false;
      },
    };
  }

  let progress = $derived(duration > 0 && isFinite(duration) ? (currentTime / duration) * 100 : 0);
</script>

<audio
  bind:this={audio}
  src={src}
  bind:paused={paused}
  bind:currentTime={currentTime}
  bind:duration={duration}
  class="hidden"
></audio>

<div class="flex items-center gap-2 h-10 px-2 rounded-md bg-slate-100 dark:bg-slate-800 select-none min-w-40">
  <!-- Play / Pause button -->
  <button
    onclick={togglePlay}
    class="flex-none flex items-center justify-center w-7 h-7 rounded-full bg-slate-700 dark:bg-slate-200 text-white dark:text-slate-800 hover:bg-slate-600 dark:hover:bg-slate-300 transition-colors"
    aria-label={paused ? "Play" : "Pause"}
  >
    {#if paused}
      <IconPlay class="w-3.5 h-3.5" />
    {:else}
      <IconPause class="w-3.5 h-3.5" />
    {/if}
  </button>

  <!-- Time -->
  <span class="flex-none text-xs text-slate-500 dark:text-slate-400 tabular-nums">
    {formatTime(currentTime)} / {formatTime(duration)}
  </span>

  <!-- Progress bar -->
  <!-- svelte-ignore a11y_no_static_element_interactions -->
  <div
    bind:this={progressBar}
    class="relative flex-1 h-full flex items-center"
    use:interactionHandler={{ click: seekFromCursor, drag: startDrag }}
    role="slider"
    aria-valuenow={currentTime}
    aria-valuemin={0}
    aria-valuemax={duration}
    tabindex={0}
  >
    <div class="relative w-full h-1.5 rounded-full bg-slate-300 dark:bg-slate-600">
      <div
        class="absolute inset-y-0 left-0 rounded-full bg-slate-700 dark:bg-slate-200 transition-[width] {dragging
          ? 'duration-0'
          : 'duration-100'}"
        style:width="calc({progress / 100} * (100% - 0.75rem) + 0.375rem)"
      ></div>
      <div
        class="absolute top-1/2 -translate-y-1/2 w-3 h-3 rounded-full bg-slate-700 dark:bg-slate-200 transition-[left] {dragging
          ? 'duration-0'
          : 'duration-100'}"
        style:left="calc({progress / 100} * (100% - 0.75rem))"
      ></div>
    </div>
  </div>
</div>
