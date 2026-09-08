<script lang="ts">
  import { onMount } from "svelte";

  import { probeWebGPU, type WebGPUStatus } from "../webgpu_check.js";

  let status: WebGPUStatus | null = $state(null);
  let dismissed: boolean = $state(false);

  onMount(() => {
    probeWebGPU().then((s) => (status = s));
  });

  // Remember across sessions via localStorage, scoped by reason so that
  // a later WebGPU-enabled environment re-surfaces the banner.
  function storageKey(s: WebGPUStatus): string {
    return `embedding-atlas:webgpu-banner-dismissed:${s.kind}:${"reason" in s ? s.reason : ""}`;
  }

  $effect(() => {
    if (status == null || status.kind === "ok") {
      dismissed = true;
      return;
    }
    try {
      dismissed = localStorage.getItem(storageKey(status)) === "1";
    } catch {
      dismissed = false;
    }
  });

  function dismiss() {
    if (status == null) return;
    try {
      localStorage.setItem(storageKey(status), "1");
    } catch {
      /* ignore */
    }
    dismissed = true;
  }
</script>

{#if status && status.kind !== "ok" && !dismissed}
  <div class="wgb" role="status" aria-live="polite">
    <span class="wgb-icon" aria-hidden="true">!</span>
    <div class="wgb-body">
      <strong>WebGPU unavailable.</strong>
      The embedding view will fall back to WebGL and point maps may be slower.
      <span class="wgb-reason">({status.reason})</span>
    </div>
    <button class="wgb-dismiss" onclick={dismiss} aria-label="Dismiss">×</button>
  </div>
{/if}

<style>
  .wgb {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.5rem 0.75rem;
    background: #fef3c7;
    color: #78350f;
    border-bottom: 1px solid #fde68a;
    font-size: 0.82rem;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }
  :global(html.dark) .wgb,
  :global(body.dark) .wgb {
    background: #422006;
    color: #fde68a;
    border-bottom-color: #713f12;
  }
  .wgb-icon {
    width: 1.2rem;
    height: 1.2rem;
    border-radius: 50%;
    background: #d97706;
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: bold;
    flex: 0 0 auto;
  }
  .wgb-body {
    flex: 1;
  }
  .wgb-reason {
    opacity: 0.7;
    margin-left: 0.25rem;
  }
  .wgb-dismiss {
    background: transparent;
    border: none;
    color: inherit;
    font-size: 1.2rem;
    cursor: pointer;
    padding: 0 0.25rem;
    opacity: 0.6;
  }
  .wgb-dismiss:hover {
    opacity: 1;
  }
</style>
