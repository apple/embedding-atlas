<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { deepMemo, interactionHandler, type CursorValue } from "@embedding-atlas/utils";
  import { flip } from "svelte/animate";
  import { scale } from "svelte/transition";

  import NewTabView from "../layouts/NewTabView.svelte";
  import PopupButton from "../widgets/PopupButton.svelte";

  import { IconClose, IconDashboardLayout, IconListLayout, IconPlus } from "../assets/icons.js";

  import type { LayoutSpec } from "../layouts/layout.js";
  import { getStoreContext } from "../stores/embedding_atlas_store.js";

  const store = getStoreContext();
  const { currentLayout, layouts, layoutOrder } = store;

  let popupVisible: boolean = $state(false);
  let renamingId: string | null = $state(null);

  let draggedId: string | null = $state(null);
  let draggingTabOrder: string[] | null = $state.raw(null);
  let dragX: number = $state(0);
  let dragY: number = $state(0);

  let displayOrder = $derived.by(deepMemo(() => draggingTabOrder ?? $layoutOrder));

  let tabElements: Map<string, HTMLElement> = new Map();

  function registerTab(element: HTMLElement, id: string) {
    tabElements.set(id, element);
    return {
      destroy() {
        tabElements.delete(id);
      },
    };
  }

  function closeTab(id: string) {
    store.removeLayout(id);
  }

  function startRename(id: string) {
    renamingId = id;
  }

  function commitRename(id: string, name: string) {
    const trimmed = name.trim();
    if (trimmed) {
      store.updateLayout<LayoutSpec>(id, (draft) => {
        draft.name = trimmed;
      });
    }
    renamingId = null;
  }

  function cancelRename() {
    renamingId = null;
  }

  function initializeRenameInput(el: HTMLInputElement) {
    el.select();
    el.addEventListener("mousedown", (e) => e.stopPropagation());
  }

  function dragOrderComputer(): (clientX: number) => string[] {
    const order = [...$layoutOrder];
    const fromIndex = order.indexOf(draggedId!);

    // Snapshot centers at drag start
    const centers: number[] = [];
    for (let i = 0; i < order.length; i++) {
      const el = tabElements.get(order[i]);
      if (!el) continue;
      const rect = el.getBoundingClientRect();
      centers.push(rect.left + rect.width / 2);
    }

    return (clientX: number) => {
      // Find insertion point using snapshotted centers
      let insertAt = centers.length;
      for (let i = 0; i <= centers.length; i++) {
        const c1 = i >= 1 ? centers[i - 1] : -Infinity;
        const c2 = i < centers.length ? centers[i] : Infinity;
        if (clientX >= c1 && clientX <= c2) {
          insertAt = i;
          break;
        }
      }

      if (insertAt === fromIndex || insertAt === fromIndex + 1) {
        return order;
      }

      const result = [...order];
      result.splice(fromIndex, 1);
      const toIndex = insertAt > fromIndex ? insertAt - 1 : insertAt;
      result.splice(toIndex, 0, draggedId!);
      return result;
    };
  }

  function startDrag(id: string) {
    return (start: CursorValue) => {
      draggedId = id;
      dragX = start.clientX;
      dragY = start.clientY;

      const computeOrder = dragOrderComputer();

      draggingTabOrder = computeOrder(start.clientX);

      return {
        move(v: CursorValue) {
          dragX = v.clientX;
          dragY = v.clientY;
          draggingTabOrder = computeOrder(v.clientX);
        },
        up() {
          draggedId = null;
          if (draggingTabOrder) {
            store.setLayoutOrder(draggingTabOrder);
            draggingTabOrder = null;
          }
        },
        cancel() {
          draggedId = null;
          draggingTabOrder = null;
        },
      };
    };
  }
</script>

<div class="flex select-none items-center gap-1">
  <div class="flex items-center select-none flex-1 overflow-hidden bg-slate-100 dark:bg-slate-900 rounded-md">
    {#each displayOrder as id (id)}
      {@const displayName = $layouts[id]?.name ?? "(none)"}
      {@const isSelected = $currentLayout === id}
      {@const isDragging = draggedId === id}
      <div class="flex items-center" animate:flip={{ duration: 200 }} transition:scale={{ duration: 200 }}>
        <button
          class="group relative flex items-center gap-0.5 px-2 py-1 h-[28px] border
          {isSelected
            ? 'z-10 text-slate-900 bg-white border-slate-300 dark:text-slate-100 dark:bg-slate-800 dark:border-slate-600'
            : 'bg-transparent border-transparent text-slate-500 dark:text-slate-400 hover:text-slate-700 hover:border-slate-100 dark:hover:text-slate-300 hover:bg-white dark:hover:bg-slate-800 dark:hover:border-slate-900'}
          {isDragging ? 'opacity-20' : ''}
          rounded-md cursor-default"
          title={displayName}
          use:registerTab={id}
          use:interactionHandler={{
            drag: startDrag(id),
          }}
          onclick={() => {
            if (renamingId !== id) {
              store.setCurrentLayout(id);
            }
          }}
        >
          {#if $layouts[id].type == "list"}
            <IconListLayout class="w-5 h-5" />
          {:else}
            <IconDashboardLayout class="w-5 h-5" />
          {/if}
          {#if renamingId === id}
            <!-- svelte-ignore a11y_autofocus -->
            <input
              type="text"
              class="mx-0.5 bg-transparent outline-none border-b border-blue-500 border-t border-t-transparent w-25"
              value={displayName}
              autofocus
              use:initializeRenameInput
              onblur={(e) => commitRename(id, e.currentTarget.value)}
              onkeydown={(e) => {
                e.stopPropagation();
                if (e.key === "Enter") commitRename(id, e.currentTarget.value);
                if (e.key === "Escape") cancelRename();
              }}
            />
          {:else}
            <!-- svelte-ignore a11y_no_static_element_interactions -->
            <span
              class="mx-0.5 text-nowrap"
              ondblclick={(e) => {
                e.stopPropagation();
                startRename(id);
              }}
            >
              {displayName}
            </span>
          {/if}

          <!-- svelte-ignore a11y_no_static_element_interactions -->
          <span
            class="p-1 flex items-center justify-center rounded-full opacity-0 group-hover:opacity-100 hover:bg-slate-200 dark:hover:bg-slate-700 cursor-pointer"
            title="Close tab"
            role="button"
            tabindex="-1"
            onclick={(e) => {
              e.stopPropagation();
              closeTab(id);
            }}
            onkeydown={(e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.stopPropagation();
                closeTab(id);
              }
            }}
          >
            <IconClose class="w-3.5 h-3.5" />
          </span>
        </button>
      </div>
    {/each}
  </div>
  <PopupButton icon={IconPlus} title="Add dashboard" placement="bottom-start" bind:visible={popupVisible}>
    {#snippet button({ toggle })}
      <button
        class="p-2 rounded-md border text-slate-500 dark:text-slate-400 border-slate-200 dark:border-slate-800 hover:bg-slate-100 hover:border-slate-100 dark:hover:bg-slate-700 dark:hover:border-slate-700"
        onclick={toggle}><IconPlus class="w-4 h-4" /></button
      >
    {/snippet}
    <NewTabView onCreate={() => (popupVisible = false)} />
  </PopupButton>
</div>

{#if draggedId}
  <div
    class="fixed pointer-events-none z-50 flex items-center gap-0.5 px-2 py-1 h-[28px] text-sm bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-600 rounded-md shadow-lg opacity-80"
    style:left="{dragX}px"
    style:top="{dragY}px"
    style:transform="translate(-50%, -50%)"
  >
    <IconDashboardLayout class="w-5 h-5" />
    <span class="mx-0.5 text-nowrap">{$layouts[draggedId]?.name ?? "(none)"}</span>
  </div>
{/if}
