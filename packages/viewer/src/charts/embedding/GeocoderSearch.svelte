<!--
  Geocoder search component using Photon (Komoot) API.
  Provides location search with autocomplete results.
  When a result is selected, emits the coordinates for map navigation.
-->
<script lang="ts">
  import { IconClose } from "../../assets/icons.js";

  interface Props {
    onSelect: (lon: number, lat: number, name: string) => void;
  }

  let { onSelect }: Props = $props();

  let query = $state("");
  let results = $state<PhotonFeature[]>([]);
  let isOpen = $state(false);
  let selectedIndex = $state(-1);
  let inputEl: HTMLInputElement | undefined = $state();
  let debounceTimer: ReturnType<typeof setTimeout> | undefined;

  interface PhotonFeature {
    geometry: { coordinates: [number, number] };
    properties: {
      name?: string;
      city?: string;
      state?: string;
      country?: string;
      osm_key?: string;
      osm_value?: string;
      type?: string;
      extent?: [number, number, number, number];
    };
  }

  function formatResult(f: PhotonFeature): string {
    const p = f.properties;
    const parts: string[] = [];
    if (p.name) parts.push(p.name);
    if (p.city && p.city !== p.name) parts.push(p.city);
    if (p.state) parts.push(p.state);
    if (p.country) parts.push(p.country);
    return parts.join(", ");
  }

  function formatType(f: PhotonFeature): string {
    const p = f.properties;
    if (p.osm_value) {
      return p.osm_value.replace(/_/g, " ");
    }
    if (p.type) return p.type;
    return "";
  }

  async function search(q: string) {
    if (q.length < 2) {
      results = [];
      isOpen = false;
      return;
    }
    try {
      const res = await fetch(`https://photon.komoot.io/api/?q=${encodeURIComponent(q)}&limit=5`);
      if (!res.ok) return;
      const data = await res.json();
      results = data.features ?? [];
      isOpen = results.length > 0;
      selectedIndex = -1;
    } catch {
      // network error — silently ignore
    }
  }

  function onInput() {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => search(query), 300);
  }

  function selectResult(feature: PhotonFeature) {
    const [lon, lat] = feature.geometry.coordinates;
    const name = formatResult(feature);
    query = name;
    isOpen = false;
    results = [];
    onSelect(lon, lat, name);
  }

  function onKeydown(e: KeyboardEvent) {
    if (!isOpen || results.length === 0) return;

    if (e.key === "ArrowDown") {
      e.preventDefault();
      selectedIndex = Math.min(selectedIndex + 1, results.length - 1);
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      selectedIndex = Math.max(selectedIndex - 1, 0);
    } else if (e.key === "Enter" && results.length > 0) {
      e.preventDefault();
      e.stopPropagation();
      selectResult(results[selectedIndex >= 0 ? selectedIndex : 0] ?? results[0]);
    } else if (e.key === "Escape") {
      isOpen = false;
    }
  }

  function clear() {
    query = "";
    results = [];
    isOpen = false;
    inputEl?.focus();
  }
</script>

<div class="relative w-56">
  <div class="relative">
    <input
      bind:this={inputEl}
      bind:value={query}
      oninput={onInput}
      onkeydown={onKeydown}
      onfocus={() => {
        if (results.length > 0) isOpen = true;
      }}
      onblur={() => {
        setTimeout(() => (isOpen = false), 200);
      }}
      type="text"
      placeholder="Go to place..."
      class="w-full text-xs rounded-md py-1 pl-2 pr-7 bg-white/75 dark:bg-slate-800/75 backdrop-blur-sm border border-slate-300 dark:border-slate-600 text-slate-800 dark:text-slate-200 placeholder:text-slate-400 dark:placeholder:text-slate-500 focus:outline-2 focus:outline-blue-600 focus:-outline-offset-1"
    />
    {#if query.length > 0}
      <button
        onclick={clear}
        class="absolute right-1 top-1/2 -translate-y-1/2 text-slate-400 dark:text-slate-500 hover:text-slate-600 dark:hover:text-slate-300"
        tabindex={-1}
      >
        <IconClose class="w-4 h-4" />
      </button>
    {/if}
  </div>

  {#if isOpen && results.length > 0}
    <ul
      class="absolute w-full mt-1 rounded-md border border-slate-300 dark:border-slate-600 bg-white/90 dark:bg-slate-800/90 backdrop-blur-sm shadow-lg z-30 overflow-hidden"
    >
      {#each results as feature, i}
        <li>
          <button
            class="w-full text-left px-2 py-1.5 text-xs cursor-pointer transition-colors hover:bg-blue-100 dark:hover:bg-blue-900"
            class:bg-blue-100={selectedIndex === i}
            class:dark:bg-blue-900={selectedIndex === i}
            onmousedown={(e) => {
              e.preventDefault();
              selectResult(feature);
            }}
          >
            <div class="text-slate-800 dark:text-slate-200 truncate">
              {formatResult(feature)}
            </div>
            {#if formatType(feature)}
              <div class="text-slate-400 dark:text-slate-500 text-[10px] truncate">
                {formatType(feature)}
              </div>
            {/if}
          </button>
        </li>
      {/each}
    </ul>
  {/if}
</div>
