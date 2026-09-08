<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import HoverTooltip from "../../widgets/HoverTooltip.svelte";

  import type { PerClassScore, Predict } from "./prediction.js";

  interface Props {
    predict: Predict;
    /** Max strength (debiased MI) across the list, for length scaling. */
    maxStrength: number;
    /** Pixel width of the SVG. */
    width: number;
    /** Bar color (used for the bar, dot, and tooltip mini-bars). */
    markColor: string;
    /** Pixel height of the SVG. Default 16 to match BinaryPredictBar. */
    height?: number;
  }

  let { predict, maxStrength, width, markColor, height = 16 }: Props = $props();

  // Visual constants in pixels. Match BinaryPredictBar's dimensions.
  const radius = 3;
  const tickHalfHeight = 4;
  const barHeight = 3;

  let cy = $derived(height / 2);
  let strength = $derived(predict.strength);
  // Reserve `radius` on the right so the dot at the end of the bar stays
  // inside the SVG box.
  let len = $derived(Math.min((strength / Math.max(maxStrength, 1e-10)) * (width - radius), width - radius));

  // Dominant class = the one this feature most confidently pulls toward
  // (largest positive per-class score). null when no class is confident.
  let perClass = $derived(predict.perClass ?? []);
  let dominant = $derived.by<PerClassScore | null>(() => {
    let best: PerClassScore | null = null;
    for (const pc of perClass) {
      if (pc.score > 0 && (best == null || pc.score > best.score)) {
        best = pc;
      }
    }
    return best;
  });

  // Tooltip rows: classes with a non-negligible signed score, sorted strongest
  // (most positive) first, then by descending magnitude.
  let rows = $derived.by<PerClassScore[]>(() =>
    perClass.filter((pc) => Math.abs(pc.score) > 1e-6).sort((a, b) => b.score - a.score),
  );
  let maxAbsScore = $derived(rows.reduce((m, pc) => Math.max(m, Math.abs(pc.score)), 0));
</script>

<HoverTooltip>
  <svg class="flex-none block" width={width} height={height} viewBox="0 0 {width} {height}">
    <!-- Baseline -->
    <line
      x1="0"
      y1={cy}
      x2={width}
      y2={cy}
      class="stroke-slate-300 dark:stroke-slate-600"
      stroke-width="1"
      shape-rendering="crispEdges"
    />
    {#if strength > 0 && len > 0}
      <!-- Bar from left (zero) to dot -->
      <rect x="0" y={cy - barHeight / 2} width={len} height={barHeight} fill={markColor} />
      <!-- Dot at the end of the bar -->
      <circle cx={len} cy={cy} r={radius} fill={markColor} />
    {/if}
    <!-- Zero tick at the left edge. Offset by 0.5 so the 1px stroke renders
         fully inside the viewBox. -->
    <line
      x1="0.5"
      y1={cy - tickHalfHeight}
      x2="0.5"
      y2={cy + tickHalfHeight}
      class="stroke-slate-500 dark:stroke-slate-400"
      stroke-width="1"
    />
  </svg>

  {#snippet content()}
    {@const sliderW = 96}
    {@const sliderH = 16}
    {@const sliderCx = sliderW / 2}
    <div class="flex flex-col gap-1.5">
      {#if dominant != null}
        <div class="flex items-center gap-1.5 font-medium">
          <span class="inline-block w-2 h-2 rounded-full" style:background-color={markColor}></span>
          <span>Predicts {dominant.className}</span>
        </div>
      {:else}
        <div class="font-medium text-slate-500 dark:text-slate-400">No confident prediction</div>
      {/if}

      {#if rows.length > 0}
        <!-- Per-class signed scores as center-tick diverging sliders (matching
             BinaryPredictBar): positive (toward class) extends right with a dot,
             negative (away) extends left. -->
        <div class="grid grid-cols-[auto_auto_auto] items-center gap-x-2 gap-y-0.5">
          {#each rows as pc (pc.className)}
            {@const goesRight = pc.score > 0}
            {@const len = Math.min(
              (Math.abs(pc.score) / Math.max(maxAbsScore, 1e-10)) * (sliderCx - radius),
              sliderCx - radius,
            )}
            <div class="truncate max-w-[8rem]">{pc.className}</div>
            <svg width={sliderW} height={sliderH} viewBox="0 0 {sliderW} {sliderH}" class="flex-none block">
              <line
                x1="0"
                y1={sliderH / 2}
                x2={sliderW}
                y2={sliderH / 2}
                class="stroke-slate-300 dark:stroke-slate-600"
                stroke-width="1"
              />
              {#if len > 0}
                <rect
                  x={goesRight ? sliderCx : sliderCx - len}
                  y={sliderH / 2 - barHeight / 2}
                  width={len}
                  height={barHeight}
                  fill={markColor}
                />
                <circle cx={goesRight ? sliderCx + len : sliderCx - len} cy={sliderH / 2} r={radius} fill={markColor} />
              {/if}
              <line
                x1={sliderCx}
                y1={sliderH / 2 - tickHalfHeight}
                x2={sliderCx}
                y2={sliderH / 2 + tickHalfHeight}
                class="stroke-slate-500 dark:stroke-slate-400"
                stroke-width="1"
              />
            </svg>
            <div class="text-right tabular-nums text-slate-700 dark:text-slate-200">
              {pc.score > 0 ? "+" : ""}{pc.score.toFixed(2)}
            </div>
          {/each}
        </div>
      {/if}

      <dl class="grid grid-cols-[auto_auto] gap-x-3 gap-y-0.5">
        <dt class="font-medium text-slate-600 dark:text-slate-300">Strength (MI)</dt>
        <dd class="text-right tabular-nums text-slate-700 dark:text-slate-200">{strength.toFixed(3)}</dd>
        {#if predict.support != null}
          <dt class="font-medium text-slate-600 dark:text-slate-300">Support</dt>
          <dd class="text-right tabular-nums text-slate-700 dark:text-slate-200">{predict.support.toLocaleString()}</dd>
        {/if}
      </dl>
      <p class="text-slate-500 dark:text-slate-400 max-w-[16rem]">
        Strength is the debiased mutual information between the feature and the class column. Per-class scores are
        confidence-aware log-odds: positive pulls toward the class, negative away.
      </p>
    </div>
  {/snippet}
</HoverTooltip>
