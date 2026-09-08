<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { onDestroy } from "svelte";

  import CheckBox from "../../widgets/CheckBox.svelte";
  import MultiSelect from "../../widgets/MultiSelect.svelte";
  import Select from "../../widgets/Select.svelte";
  import Container from "../common/Container.svelte";
  import FeatureRow from "./FeatureRow.svelte";

  import type { ChartViewProps } from "../chart.js";
  import { resolveChartTheme, type ChartTheme } from "../common/theme.js";
  import { FeaturesListStore, makeSortFunc, type ListItem } from "./features_list_store.js";
  import type { FeaturesListSpec, FeaturesListState } from "./types.js";

  let {
    context,
    width,
    height,
    spec,
    state: chartState,
    onStateChange,
    onSpecChange,
  }: ChartViewProps<FeaturesListSpec, FeaturesListState> = $props();

  // svelte-ignore state_referenced_locally
  let { colorScheme, theme: themeConfig } = context;
  let theme = $derived(resolveChartTheme($colorScheme, $themeConfig));

  // svelte-ignore state_referenced_locally
  const store = new FeaturesListStore(context, spec, chartState, () =>
    onStateChange((d) => {
      delete d.selected;
    }),
  );

  $effect.pre(() => {
    store.update(spec, chartState);
  });

  onDestroy(() => store.destroy());

  const data = store.data;
  const allSources = store.allSources;
  const topicsByFeature = store.topicsByFeature;

  let listWidth = $state.raw(400);

  let hasPredict = $derived(spec.data.predict != null);
  let predictName = $derived(typeof spec.data.predict === "string" ? spec.data.predict : "expression");
  let activeSourceCount = $derived($allSources?.length ?? 0);

  // "Show more / less" toggle: expand the limit to 500, or collapse back to 100.
  const EXPANDED_LIMIT = 500;
  const DEFAULT_LIMIT = 100;
  let featureLimit = $derived(spec.limit ?? DEFAULT_LIMIT);

  // --- Selection / pinning state (lives in the chart view state) ---
  let selected = $derived(chartState.selected ?? []);
  let pinned = $derived(chartState.pinned ?? []);
  let selectedSet = $derived(new Set(selected));
  let pinnedSet = $derived(new Set(pinned));

  // --- Presentation pipeline (search → sort → limit → group), applied here so
  // pinned features can stay visible outside it. ---
  let allItems = $derived($data.allItems);
  let byFeature = $derived.by(() => {
    const m = new Map<string, ListItem>();
    for (const it of allItems) {
      m.set(it.feature, it);
    }
    return m;
  });

  let searched = $derived.by(() => {
    const term = (chartState.search ?? "").trim().toLowerCase();
    if (term.length === 0) {
      return allItems;
    }
    return allItems.filter((it) => it.feature.toLowerCase().includes(term));
  });

  let sorted = $derived.by(() => {
    const arr = searched.slice();
    arr.sort(makeSortFunc(spec.sort, hasPredict, $data.sourceCount));
    return arr;
  });

  let limited = $derived(sorted.slice(0, featureLimit));
  let shownCount = $derived(limited.length);
  // True when the limit truncated the matched set — raising it reveals more.
  let hasMore = $derived(sorted.length > limited.length);
  let canShowLess = $derived(featureLimit >= EXPANDED_LIMIT);
  let canShowMore = $derived(!canShowLess && hasMore);

  // Pinned features in pin order, looked up from the (unfiltered) item set.
  // Features absent from the aggregation (e.g. filtered out by source) are skipped.
  let pinnedItems = $derived.by(() => {
    const out: ListItem[] = [];
    for (const f of pinned) {
      const it = byFeature.get(f);
      if (it != null) {
        out.push(it);
      }
    }
    return out;
  });

  type Group = { name: string; items: ListItem[] };
  let groups = $derived.by<Group[]>(() => {
    if (spec.groupBy === "topics" && $topicsByFeature) {
      const topics = $topicsByFeature;
      const map = new Map<string, ListItem[]>();
      const order: string[] = [];
      const push = (name: string, it: ListItem) => {
        let arr = map.get(name);
        if (arr == null) {
          arr = [];
          map.set(name, arr);
          order.push(name);
        }
        arr.push(it);
      };
      for (const it of limited) {
        const ts = topics.get(it.feature);
        if (ts && ts.length > 0) {
          for (const t of ts) {
            push(t, it);
          }
        } else {
          push("Other", it);
        }
      }
      return order.map((name) => ({ name, items: map.get(name)! }));
    }
    return [{ name: "Features", items: limited }];
  });
  let topicCount = $derived(spec.groupBy === "topics" ? groups.length : 0);

  const sourceOptions = $derived.by(() => {
    const sources = $allSources ?? [];
    return sources.map((s) => ({ label: s, value: s }));
  });

  // spec.sources === undefined means "no filter" → all sources are conceptually
  // selected. Map that to a fully-checked list for the MultiSelect.
  const sourceValues = $derived(spec.sources ?? $allSources ?? []);

  const predictOptions = $derived.by(() => {
    const opts: Array<{ label: string; value: any; disabled?: boolean } | "---"> = [
      { label: "(none)", value: undefined },
    ];
    const cols = (context.tables[context.table]?.columns ?? []).filter(
      (c) => c.jsType != null && c.name !== context.id && c.name !== spec.data.features,
    );
    if (cols.length > 0) {
      opts.push("---");
      for (const c of cols) {
        opts.push({ label: c.name, value: c.name });
      }
    }
    return opts;
  });

  const sortOptions = $derived.by(() => {
    const sourceCount = $allSources?.length ?? 0;
    const moreThanOneSource = sourceCount >= 2;
    return [
      { label: "↓ Frequency", value: "frequency-descending" },
      { label: "↑ Frequency", value: "frequency-ascending" },
      "---" as const,
      { label: "↓ Predictiveness", value: "predictiveness-descending", disabled: !hasPredict },
      { label: "↑ Predictiveness", value: "predictiveness-ascending", disabled: !hasPredict },
      "---" as const,
      { label: "↓ Source skew", value: "source-skew-descending", disabled: !moreThanOneSource },
      { label: "↑ Source skew", value: "source-skew-ascending", disabled: !moreThanOneSource },
      "---" as const,
      { label: "↓ A-Z", value: "alphabetical" },
    ];
  });

  let segmentColors = $derived.by(() => buildSegmentColors($allSources ?? [], theme));

  let isBinary = $derived(hasPredict && $data.classNames.length === 2);
  let directionColors = $derived(pickDirectionColors(theme, $data.hasClassPolarity));

  // Bar scales: max over what's actually shown (pinned section + main list) so
  // bars are comparable across both.
  let maxCount = $derived.by(() => {
    let m = 0;
    for (const it of pinnedItems) m = Math.max(m, it.count);
    for (const it of limited) m = Math.max(m, it.count);
    return m;
  });

  let maxStrength = $derived.by(() => {
    if (!hasPredict) return 0;
    let m = 0;
    for (const it of pinnedItems) m = Math.max(m, Math.abs(it.predict?.strength ?? 0));
    for (const it of limited) m = Math.max(m, Math.abs(it.predict?.strength ?? 0));
    return m;
  });

  function buildSegmentColors(sources: string[], theme: ChartTheme): Record<string, string> {
    let palette: string[];
    if (typeof theme.categoryColors === "function") {
      palette = theme.categoryColors(Math.max(sources.length, 1));
    } else {
      palette = theme.categoryColors;
    }
    const out: Record<string, string> = {};
    sources.forEach((s, i) => {
      out[s] = palette[i % palette.length];
    });
    return out;
  }

  function pickDirectionColors(t: ChartTheme, polarity: boolean): { left: string; right: string } {
    if (polarity) {
      // Diverging red / blue — colorblind-friendlier than red/green.
      // left = classNames[0] (negative), right = classNames[1] (positive).
      return { left: "#ef4444", right: "#3b82f6" };
    }
    // No semantic polarity: take two distinct hues from the category palette so
    // direction is visible without implying good/bad.
    const palette = typeof t.categoryColors === "function" ? t.categoryColors(2) : t.categoryColors;
    return { left: palette[0] ?? t.markColor, right: palette[1] ?? t.markColor };
  }

  function countLabel(item: ListItem): string {
    if (!$data.hasSelection) {
      return item.count.toLocaleString();
    }
    return `${item.countSelected.toLocaleString()} / ${item.count.toLocaleString()}`;
  }

  /** Class this feature most confidently predicts toward, or null. Multi-class only. */
  function dominantClass(item: ListItem): string | null {
    let best: { className: string; score: number } | null = null;
    for (const pc of item.predict?.perClass ?? []) {
      if (pc.score > 0 && (best == null || pc.score > best.score)) {
        best = pc;
      }
    }
    return best?.className ?? null;
  }

  function tooltip(item: ListItem): string {
    const lines: string[] = [item.feature];
    if (hasPredict) {
      const p = item.predict;
      if (p?.direction !== undefined) {
        // `direction` is always set for binary predict, even at strength 0; only
        // claim a prediction when the direction is confident (matches BinaryPredictBar).
        if (p.strength > 0) {
          lines.push(`Predicts: ${$data.classNames[p.direction]}`);
        } else {
          lines.push("No confident prediction");
        }
        lines.push(`Strength: ${p.strength.toFixed(2)}`);
        if (p.phi !== undefined) {
          lines.push(`φ: ${p.phi.toFixed(3)}`);
        }
      } else {
        const dom = dominantClass(item);
        lines.push(dom != null ? `Predicts: ${dom}` : "No confident prediction");
        lines.push(`Strength (MI): ${(p?.strength ?? 0).toFixed(3)}`);
      }
      lines.push(`Count: ${item.count.toLocaleString()}`);
    } else {
      lines.push(`Count: ${item.count.toLocaleString()}`);
      if ($data.hasSelection) {
        lines.push(`Selected: ${item.countSelected.toLocaleString()}`);
      }
    }
    if (item.sourceShape != null) {
      lines.push(`Source skew: ${item.sourceShape.toFixed(2)} (0 = balanced, 1 = single source)`);
    }
    if (item.sourceSegments && item.sourceSegments.length > 0) {
      for (const seg of item.sourceSegments) {
        lines.push(`  ${seg.source}: ${seg.count.toLocaleString()}`);
      }
    }
    return lines.join("\n");
  }

  // --- Selection / pin handlers ---
  function applySelection(next: string[]) {
    onStateChange((d) => {
      if (next.length === 0) {
        delete d.selected;
      } else {
        d.selected = next;
      }
    });
  }

  function toggleSelect(feature: string) {
    const cur = chartState.selected ?? [];
    applySelection(cur.includes(feature) ? cur.filter((x) => x !== feature) : [...cur, feature]);
  }

  function onRowClick(feature: string, shift: boolean) {
    context.textHighlight.set(feature);
    if (shift) {
      toggleSelect(feature);
    } else {
      // Plain click toggles the single-feature focus: clicking the sole selected
      // feature clears the selection; anything else collapses to just this one.
      const cur = chartState.selected ?? [];
      const onlyThis = cur.length === 1 && cur[0] === feature;
      applySelection(onlyThis ? [] : [feature]);
    }
  }

  function togglePin(feature: string) {
    const cur = chartState.pinned ?? [];
    const next = cur.includes(feature) ? cur.filter((x) => x !== feature) : [...cur, feature];
    onStateChange((d) => {
      if (next.length === 0) {
        delete d.pinned;
      } else {
        d.pinned = next;
      }
    });
  }
</script>

<Container width={width} height={height}>
  <div class="flex flex-col gap-2 select-none w-full h-full" bind:clientWidth={listWidth}>
    <!-- Search bar -->
    <input
      type="search"
      value={chartState.search ?? ""}
      placeholder="Search features"
      oninput={(e) =>
        onStateChange((d) => {
          d.search = (e.currentTarget as HTMLInputElement).value;
        })}
      class="form-input rounded-md py-1 text-sm bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600"
    />

    <!-- Source and predict -->
    <div class="flex gap-2">
      <div class="flex-1 flex flex-col gap-0.5">
        <div class="text-xs font-medium uppercase text-slate-400">Source</div>
        <MultiSelect
          values={sourceValues}
          options={sourceOptions}
          class="w-full"
          placeholder="No sources"
          onChange={(v) =>
            onSpecChange((d) => {
              // "All selected" collapses to undefined (no filter), matching
              // the existing semantics of spec.sources.
              d.sources = v.length === sourceOptions.length ? undefined : v;
            })}
        />
      </div>
      <div class="flex-1 flex flex-col gap-0.5">
        <div class="text-xs font-medium uppercase text-slate-400">Predict</div>
        <Select
          value={spec.data.predict}
          options={predictOptions}
          class="w-full"
          onChange={(v) =>
            onSpecChange((d) => {
              d.data.predict = v;
              if (v != null) {
                d.sort = "predictiveness-descending";
                delete d.segmentBy;
              } else {
                if (d.sort == "predictiveness-ascending" || d.sort == "predictiveness-descending") {
                  d.sort = "frequency-descending";
                }
              }
            })}
        />
      </div>
    </div>

    <!-- Sort and grouping -->
    <div class="flex gap-2">
      <div class="flex-1 flex flex-col gap-0.5">
        <div class="text-xs font-medium uppercase text-slate-400">Order</div>
        <Select
          value={spec.sort ?? "frequency-descending"}
          options={sortOptions}
          onChange={(v) =>
            onSpecChange((d) => {
              d.sort = v;
            })}
        />
      </div>
      <div class="flex-1 flex flex-col gap-0.5">
        <div class="text-xs font-medium uppercase text-slate-400">Group</div>
        <div class="flex gap-2">
          {#if spec.data.predict == null}
            <CheckBox
              label="Source"
              bind:checked={
                () => spec.segmentBy == "source",
                (v) => {
                  onSpecChange((d) => {
                    d.segmentBy = v ? "source" : undefined;
                  });
                }
              }
            />
          {/if}
          {#if spec.metadata?.topics != null}
            <CheckBox
              label="Topics"
              bind:checked={
                () => spec.groupBy == "topics",
                (v) => {
                  onSpecChange((d) => {
                    d.groupBy = v ? "topics" : undefined;
                  });
                }
              }
            />
          {/if}
        </div>
      </div>
    </div>

    <!-- Basic stats -->
    <div class="text-slate-500 dark:text-slate-400 text-xs">
      {activeSourceCount} source{activeSourceCount === 1 ? "" : "s"}
      ·
      {hasPredict ? `predicting ${predictName}` : "count"}
      ·
      {shownCount} / {$data.totalCount} features
      {#if spec.groupBy === "topics"}
        ·
        {topicCount} topic{topicCount === 1 ? "" : "s"}
      {/if}
      {#if selected.length > 0}
        ·
        {selected.length} selected
      {/if}
    </div>

    <!-- Sources color legend -->
    {#if spec.segmentBy === "source" && !hasPredict && ($allSources?.length ?? 0) > 0}
      <div class="flex flex-wrap gap-x-3 gap-y-1 text-sm text-slate-500 dark:text-slate-400">
        {#each $allSources ?? [] as src (src)}
          <span class="inline-flex items-center gap-1">
            <span
              class="inline-block w-2.5 h-2.5 rounded-sm"
              style:background-color={segmentColors[src] ?? theme.markColor}
            ></span>
            {src}
          </span>
        {/each}
      </div>
    {/if}

    <!-- List of features -->
    <div class="flex-1 w-full overflow-x-hidden overflow-y-scroll">
      <!-- Single-grid layout: pin/select buttons | name | count text | bar | strength.
           - Count mode: bar = count_bar (segmented when segmentBy=source).
           - Predict mode: bar = predict_bar (binary or G), regardless of segmentBy.
           Pinned features render in a sticky section at the top (bypassing search,
           sort, and limit) and still appear in the main list below. One grid means
           the bar column auto-aligns across both sections and topic boundaries. -->
      <div class="grid grid-cols-[max-content_minmax(0,1fr)_max-content_max-content_max-content] gap-x-2 items-center">
        <!-- Header + pinned block stay stuck at the top while the list scrolls. The
             wrapper is a chained subgrid (not a separate grid) so its columns keep
             inheriting the parent's tracks — header labels and bars stay aligned with
             the main list. An opaque background + z-index lets scrolled rows pass
             cleanly underneath. -->
        <div class="col-span-5 grid grid-cols-subgrid items-center sticky top-0 z-10 bg-white dark:bg-black">
          <!-- Column header: bulk pin/select actions in the leading column, then
               labels that line up with each FeatureRow column. -->
          <div
            class="col-span-5 grid grid-cols-subgrid items-center h-[24px] text-xs font-medium uppercase text-slate-400 dark:text-slate-500 border-b border-slate-200 dark:border-slate-700"
          >
            <div></div>
            <div class="truncate">Feature</div>
            <div class="text-right">Count</div>
            <div></div>
            <div></div>
          </div>

          {#if pinnedItems.length > 0}
            {#each pinnedItems as item (item.feature)}
              <FeatureRow
                item={item}
                selected={selectedSet.has(item.feature)}
                pinned={true}
                hasPredict={hasPredict}
                isBinary={isBinary}
                maxCount={maxCount}
                maxStrength={maxStrength}
                segmented={spec.segmentBy === "source"}
                segmentColors={segmentColors}
                directionColors={directionColors}
                classNames={$data.classNames}
                markColor={theme.markColor}
                markColorFade={theme.markColorFade}
                countLabel={countLabel(item)}
                tooltip={tooltip(item)}
                onRowClick={(shift) => onRowClick(item.feature, shift)}
                onToggleSelect={() => toggleSelect(item.feature)}
                onTogglePin={() => togglePin(item.feature)}
              />
            {/each}
            <div class="col-span-5 border-b border-dashed border-slate-300 dark:border-slate-600 my-1"></div>
          {/if}
        </div>

        {#each groups as group (group.name)}
          {#if spec.groupBy === "topics"}
            <div
              class="col-span-5 font-medium not-first:mt-2 text-slate-500 dark:text-slate-300 border-b border-slate-400"
            >
              # {group.name}
            </div>
          {/if}
          {#each group.items as item (group.name + "/" + item.feature)}
            <FeatureRow
              item={item}
              selected={selectedSet.has(item.feature)}
              pinned={pinnedSet.has(item.feature)}
              hasPredict={hasPredict}
              isBinary={isBinary}
              maxCount={maxCount}
              maxStrength={maxStrength}
              segmented={spec.segmentBy === "source"}
              segmentColors={segmentColors}
              directionColors={directionColors}
              classNames={$data.classNames}
              markColor={theme.markColor}
              markColorFade={theme.markColorFade}
              countLabel={countLabel(item)}
              tooltip={tooltip(item)}
              onRowClick={(shift) => onRowClick(item.feature, shift)}
              onToggleSelect={() => toggleSelect(item.feature)}
              onTogglePin={() => togglePin(item.feature)}
            />
          {/each}
        {/each}
      </div>

      {#if canShowMore || canShowLess}
        <div class="flex justify-center py-1">
          <button
            type="button"
            class="py-0.5 text-slate-400 dark:text-slate-500 hover:text-slate-800 dark:hover:text-slate-200 whitespace-nowrap transition-colors duration-150"
            onclick={() =>
              onSpecChange((d) => {
                d.limit = canShowLess ? DEFAULT_LIMIT : EXPANDED_LIMIT;
              })}
          >
            {#if canShowLess}
              ↑ Up to {DEFAULT_LIMIT} features
            {:else}
              ↓ Up to {EXPANDED_LIMIT} features
            {/if}
          </button>
        </div>
      {/if}
    </div>
  </div>
</Container>
