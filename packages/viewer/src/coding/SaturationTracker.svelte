<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import type { CodingState, Code, CodingEvent } from "./types.js";

  interface Props {
    codingState: CodingState;
    totalDataPoints: number;
  }

  let { codingState, totalDataPoints }: Props = $props();

  // Calculate saturation metrics
  let metrics = $derived.by(() => {
    const codes = Object.values(codingState.codes);
    const events = codingState.codingEvents;

    // Get coded data points
    const codedPoints = new Set<string>();
    for (const [pointId, codeIds] of Object.entries(codingState.codeApplications)) {
      if (codeIds.length > 0) {
        codedPoints.add(pointId);
      }
    }

    // Calculate coverage
    const coverage = totalDataPoints > 0
      ? (codedPoints.size / totalDataPoints) * 100
      : 0;

    // Calculate code growth over time (last 10 coding events)
    const recentEvents = events.slice(-20);
    const newCodesInRecent = recentEvents.filter((e) => e.action === "create").length;

    // Calculate if we're reaching saturation
    // Saturation indicators:
    // 1. High coverage (>80%)
    // 2. Few new codes being created recently
    // 3. Most coding is applying existing codes

    const applyEvents = recentEvents.filter((e) => e.action === "apply").length;
    const createEvents = recentEvents.filter((e) => e.action === "create").length;

    const saturationScore = calculateSaturationScore(
      coverage,
      createEvents,
      applyEvents,
      codes.length
    );

    // Group codes by level for analysis
    const codesByLevel = {
      open: codes.filter((c) => c.level === 1),
      axial: codes.filter((c) => c.level === 2),
      selective: codes.filter((c) => c.level === 3),
    };

    // Find least used codes (potential for merging or removal)
    const underusedCodes = codes
      .filter((c) => c.frequency > 0 && c.frequency < 3)
      .sort((a, b) => a.frequency - b.frequency)
      .slice(0, 5);

    // Find most used codes
    const topCodes = codes
      .sort((a, b) => b.frequency - a.frequency)
      .slice(0, 5);

    return {
      totalCodes: codes.length,
      codedPoints: codedPoints.size,
      coverage,
      saturationScore,
      recentNewCodes: newCodesInRecent,
      codesByLevel,
      underusedCodes,
      topCodes,
    };
  });

  function calculateSaturationScore(
    coverage: number,
    recentCreates: number,
    recentApplies: number,
    totalCodes: number
  ): number {
    // Weight factors
    const coverageWeight = 0.4;
    const newCodeWeight = 0.3;
    const applyRatioWeight = 0.3;

    // Coverage score (0-100)
    const coverageScore = Math.min(coverage, 100);

    // New code score (fewer new codes = higher saturation)
    // If creating many new codes, saturation is low
    const newCodeScore = totalCodes > 0
      ? Math.max(0, 100 - (recentCreates / Math.max(totalCodes, 1)) * 500)
      : 0;

    // Apply ratio score (more applies vs creates = higher saturation)
    const totalRecent = recentCreates + recentApplies;
    const applyRatioScore = totalRecent > 0
      ? (recentApplies / totalRecent) * 100
      : 50;

    return Math.round(
      coverageScore * coverageWeight +
      newCodeScore * newCodeWeight +
      applyRatioScore * applyRatioWeight
    );
  }

  function getSaturationLabel(score: number): string {
    if (score >= 80) return "High";
    if (score >= 60) return "Moderate";
    if (score >= 40) return "Developing";
    return "Early";
  }

  function getSaturationColor(score: number): string {
    if (score >= 80) return "#22c55e"; // green
    if (score >= 60) return "#eab308"; // yellow
    if (score >= 40) return "#f97316"; // orange
    return "#ef4444"; // red
  }
</script>

<div class="saturation-tracker">
  <h4 class="text-sm font-semibold text-slate-700 dark:text-slate-300 mb-3">
    Coding Progress
  </h4>

  <!-- Saturation Score -->
  <div class="saturation-score">
    <div class="score-header">
      <span class="text-xs text-slate-500 dark:text-slate-400">Saturation</span>
      <span
        class="score-badge"
        style:background-color={getSaturationColor(metrics.saturationScore)}
      >
        {getSaturationLabel(metrics.saturationScore)}
      </span>
    </div>
    <div class="progress-bar">
      <div
        class="progress-fill"
        style:width="{metrics.saturationScore}%"
        style:background-color={getSaturationColor(metrics.saturationScore)}
      ></div>
    </div>
    <span class="text-xs text-slate-400">{metrics.saturationScore}%</span>
  </div>

  <!-- Coverage -->
  <div class="metric-row">
    <span class="metric-label">Data Coverage</span>
    <span class="metric-value">
      {metrics.codedPoints} / {totalDataPoints}
      <span class="text-slate-400">({metrics.coverage.toFixed(1)}%)</span>
    </span>
  </div>

  <!-- Codes by Level -->
  <div class="codes-by-level">
    <span class="metric-label">Codes by Level</span>
    <div class="level-bars">
      <div class="level-bar">
        <span class="level-label">Open</span>
        <div class="level-count">{metrics.codesByLevel.open.length}</div>
      </div>
      <div class="level-bar">
        <span class="level-label">Axial</span>
        <div class="level-count">{metrics.codesByLevel.axial.length}</div>
      </div>
      <div class="level-bar">
        <span class="level-label">Selective</span>
        <div class="level-count">{metrics.codesByLevel.selective.length}</div>
      </div>
    </div>
  </div>

  <!-- Recent Activity -->
  <div class="metric-row">
    <span class="metric-label">New codes (recent)</span>
    <span class="metric-value">{metrics.recentNewCodes}</span>
  </div>

  <!-- Top Codes -->
  {#if metrics.topCodes.length > 0}
    <div class="code-list">
      <span class="metric-label">Most Used</span>
      <div class="code-items">
        {#each metrics.topCodes as code}
          <div class="code-item">
            <span
              class="code-dot"
              style:background-color={code.color}
            ></span>
            <span class="code-name">{code.name}</span>
            <span class="code-freq">{code.frequency}</span>
          </div>
        {/each}
      </div>
    </div>
  {/if}

  <!-- Underused Codes -->
  {#if metrics.underusedCodes.length > 0}
    <div class="code-list warning">
      <span class="metric-label">
        Low Frequency
        <span class="hint">(consider merging)</span>
      </span>
      <div class="code-items">
        {#each metrics.underusedCodes as code}
          <div class="code-item">
            <span
              class="code-dot"
              style:background-color={code.color}
            ></span>
            <span class="code-name">{code.name}</span>
            <span class="code-freq">{code.frequency}</span>
          </div>
        {/each}
      </div>
    </div>
  {/if}

  <!-- Saturation Tips -->
  <div class="tips">
    {#if metrics.saturationScore < 40}
      <p class="tip">Keep coding to discover patterns. Create new codes as needed.</p>
    {:else if metrics.saturationScore < 60}
      <p class="tip">Good progress! Start looking for relationships between codes.</p>
    {:else if metrics.saturationScore < 80}
      <p class="tip">Consider consolidating similar codes into axial categories.</p>
    {:else}
      <p class="tip">High saturation reached. Focus on theoretical integration.</p>
    {/if}
  </div>
</div>

<style>
  .saturation-tracker {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
    padding: 0.75rem;
    background: #f8fafc;
    border-radius: 0.5rem;
  }

  :global(.dark) .saturation-tracker {
    background: #334155;
  }

  .saturation-score {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
  }

  .score-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .score-badge {
    padding: 0.125rem 0.5rem;
    font-size: 0.625rem;
    font-weight: 600;
    color: white;
    border-radius: 9999px;
  }

  .progress-bar {
    height: 6px;
    background: #e2e8f0;
    border-radius: 3px;
    overflow: hidden;
  }

  :global(.dark) .progress-bar {
    background: #475569;
  }

  .progress-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.3s ease;
  }

  .metric-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .metric-label {
    font-size: 0.75rem;
    color: #64748b;
  }

  :global(.dark) .metric-label {
    color: #94a3b8;
  }

  .metric-value {
    font-size: 0.75rem;
    font-weight: 500;
    color: #334155;
  }

  :global(.dark) .metric-value {
    color: #e2e8f0;
  }

  .codes-by-level {
    display: flex;
    flex-direction: column;
    gap: 0.375rem;
  }

  .level-bars {
    display: flex;
    gap: 0.5rem;
  }

  .level-bar {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 0.375rem;
    background: white;
    border-radius: 0.25rem;
  }

  :global(.dark) .level-bar {
    background: #1e293b;
  }

  .level-label {
    font-size: 0.625rem;
    color: #94a3b8;
  }

  .level-count {
    font-size: 1rem;
    font-weight: 600;
    color: #334155;
  }

  :global(.dark) .level-count {
    color: #e2e8f0;
  }

  .code-list {
    display: flex;
    flex-direction: column;
    gap: 0.375rem;
  }

  .code-list.warning .metric-label {
    color: #f59e0b;
  }

  .hint {
    font-weight: normal;
    font-style: italic;
  }

  .code-items {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
  }

  .code-item {
    display: flex;
    align-items: center;
    gap: 0.375rem;
    padding: 0.25rem 0.375rem;
    background: white;
    border-radius: 0.25rem;
  }

  :global(.dark) .code-item {
    background: #1e293b;
  }

  .code-dot {
    width: 0.5rem;
    height: 0.5rem;
    border-radius: 50%;
    flex-shrink: 0;
  }

  .code-name {
    flex: 1;
    font-size: 0.75rem;
    color: #475569;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  :global(.dark) .code-name {
    color: #cbd5e1;
  }

  .code-freq {
    font-size: 0.625rem;
    color: #94a3b8;
  }

  .tips {
    padding-top: 0.5rem;
    border-top: 1px dashed #e2e8f0;
  }

  :global(.dark) .tips {
    border-color: #475569;
  }

  .tip {
    font-size: 0.75rem;
    color: #64748b;
    font-style: italic;
  }

  :global(.dark) .tip {
    color: #94a3b8;
  }
</style>
