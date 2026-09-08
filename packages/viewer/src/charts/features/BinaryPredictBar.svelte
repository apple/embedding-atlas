<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import HoverTooltip from "../../widgets/HoverTooltip.svelte";

  import type { Predict } from "./prediction.js";

  interface Props {
    /** Predict info for a binary classification. `direction` must be set. */
    predict: Predict;
    /** Max strength across the list, for length scaling. */
    maxStrength: number;
    /** Pixel width of the SVG. */
    width: number;
    /** Color pair: `left` = classNames[0] (direction 0), `right` = classNames[1] (direction 1). */
    directionColors: { left: string; right: string };
    /** Class label names, ordered [classNames[0], classNames[1]]; used for the tooltip. */
    classNames?: string[];
    /** Pixel height of the SVG. Default 16 to fit the centered tick. */
    height?: number;
  }

  let { predict, maxStrength, width, directionColors, classNames, height = 16 }: Props = $props();

  // Visual constants in pixels.
  const radius = 3;
  const tickHalfHeight = 4;
  const barHeight = 3;

  let cx = $derived(width / 2);
  let cy = $derived(height / 2);
  let strength = $derived(predict.strength);
  let goesRight = $derived(predict.direction === 1);
  let len = $derived(Math.min((strength / Math.max(maxStrength, 1e-10)) * (cx - radius), cx - radius));
  let color = $derived(goesRight ? directionColors.right : directionColors.left);
  let barX = $derived(goesRight ? cx : cx - len);
  let dotX = $derived(goesRight ? cx + len : cx - len);

  let target = $derived(predict.direction != null ? classNames?.[predict.direction] : undefined);
  let hasConfidentDirection = $derived(strength > 0);
</script>

<HoverTooltip>
  <svg class="flex-none block" width={width} height={height} viewBox="0 0 {width} {height}">
    <!-- Baseline -->
    <line x1="0" y1={cy} x2={width} y2={cy} class="stroke-slate-300 dark:stroke-slate-600" stroke-width="1" />
    {#if strength > 0 && len > 0}
      <!-- Bar from center to dot -->
      <rect x={barX} y={cy - barHeight / 2} width={len} height={barHeight} fill={color} />
      <!-- Dot at the end of the bar -->
      <circle cx={dotX} cy={cy} r={radius} fill={color} />
    {/if}
    <!-- Center tick (zero) -->
    <line
      x1={cx}
      y1={cy - tickHalfHeight}
      x2={cx}
      y2={cy + tickHalfHeight}
      class="stroke-slate-500 dark:stroke-slate-400"
      stroke-width="1"
    />
  </svg>

  {#snippet content()}
    <div class="flex flex-col gap-1.5">
      {#if hasConfidentDirection && target != null}
        <div class="flex items-center gap-1.5 font-medium">
          <span class="inline-block w-2 h-2 rounded-full" style:background-color={color}></span>
          <span>Predicts <span style:color={color}>{target}</span></span>
        </div>
      {:else}
        <div class="font-medium text-slate-500 dark:text-slate-400">No confident prediction</div>
      {/if}
      <dl class="grid grid-cols-[auto_auto] gap-x-3 gap-y-0.5">
        {#if predict.phi != null}
          <dt class="font-medium text-slate-600 dark:text-slate-300">φ coefficient</dt>
          <dd class="text-right tabular-nums text-slate-700 dark:text-slate-200">{predict.phi.toFixed(3)}</dd>
        {/if}
        <dt class="font-medium text-slate-600 dark:text-slate-300">Strength</dt>
        <dd class="text-right tabular-nums text-slate-700 dark:text-slate-200">{strength.toFixed(3)}</dd>
      </dl>
      <p class="text-slate-500 dark:text-slate-400 max-w-[16rem]">
        {#if predict.phi != null}φ is the correlation between the feature being present and the item's class (0 = no
          association, ±1 = near-perfect).
        {/if}Strength is a conservative lower bound on the association (log-odds ratio) after accounting for all
        features tested; 0 means indistinguishable from chance, higher means a larger, more reliable signal.
      </p>
    </div>
  {/snippet}
</HoverTooltip>
