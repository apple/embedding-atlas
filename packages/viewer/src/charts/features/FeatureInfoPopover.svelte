<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script module lang="ts">
  import type { ListItem } from "./features_list_store.js";

  export type RelatedState =
    | { status: "loading" }
    | { status: "error"; message: string }
    | { status: "ready"; items: ListItem[] };
</script>

<script lang="ts">
  import { autoUpdate, computePosition, flip, offset, shift } from "@floating-ui/dom";
  import type { Snippet } from "svelte";

  import { IconClose } from "../../assets/icons.js";
  import Spinner from "../../widgets/Spinner.svelte";

  interface Props {
    /** Element the panel is positioned against (the clicked info button). */
    anchor: HTMLElement;
    /** The feature the panel is describing. */
    currentItem: ListItem;
    /** Resolved description text, or null when configured-but-empty / not configured. */
    description: string | null;
    /** Whether `metadata.description` is configured at all (controls the empty-state copy). */
    descriptionConfigured: boolean;
    /** Similar-features fetch state. */
    related: RelatedState;
    /** Renders one feature row (supplied by the parent so rows behave like the main list). */
    renderRow: Snippet<[ListItem]>;
    /** Grid column template shared with the main list, so row columns stay aligned. */
    gridClass: string;
    onClose: () => void;
  }

  let { anchor, currentItem, description, descriptionConfigured, related, renderRow, gridClass, onClose }: Props =
    $props();

  let panel: HTMLDivElement;

  // Position against the anchor and keep it there while the list scrolls / resizes.
  // If the anchoring row scrolls away and unmounts, close instead of floating orphaned.
  $effect(() => {
    const anchorEl = anchor;
    panel.showPopover();
    const cleanup = autoUpdate(anchorEl, panel, () => {
      if (!anchorEl.isConnected) {
        onClose();
        return;
      }
      computePosition(anchorEl, panel, {
        strategy: "fixed",
        placement: "bottom-start",
        middleware: [offset(6), flip(), shift({ padding: 8 })],
      }).then(({ x, y }) => {
        panel.style.left = `${x}px`;
        panel.style.top = `${y}px`;
      });
    });
    return () => {
      cleanup();
      try {
        panel.hidePopover();
      } catch {
        // Element may already be detached; ignore.
      }
    };
  });

  // Light dismiss. Clicks on any info trigger are ignored here so the trigger's
  // own handler can re-target the (already open) popover without a close/reopen flicker.
  $effect(() => {
    function onPointerDown(e: PointerEvent) {
      const target = e.target as Node | null;
      if (target != null && panel.contains(target)) {
        return;
      }
      if (target instanceof Element && target.closest("[data-feature-info-trigger]")) {
        return;
      }
      onClose();
    }
    function onKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape" && !e.defaultPrevented) {
        onClose();
        e.stopPropagation();
      }
    }
    // Escape listens on `document` in the bubble phase: nested dismissables
    // (Select / MultiSelect / PopupButton) handle Escape on their own element and
    // stop propagation, so they close first without closing this panel; and we
    // still run before the window-level "reset filters" handler, which we suppress.
    window.addEventListener("pointerdown", onPointerDown, true);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      window.removeEventListener("pointerdown", onPointerDown, true);
      document.removeEventListener("keydown", onKeyDown);
    };
  });
</script>

<div
  bind:this={panel}
  popover="manual"
  class="fixed m-0 z-30 w-[380px] max-w-[calc(100vw-1rem)] max-h-[70vh] overflow-y-auto rounded-md p-3 text-slate-700 dark:text-slate-200 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-600 shadow-lg select-none"
>
  <div class="flex items-start justify-between gap-2">
    <div class="font-medium break-words min-w-0" title={currentItem.feature}>{currentItem.feature}</div>
    <button
      type="button"
      class="shrink-0 text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 transition-colors duration-150"
      title="Close"
      onclick={onClose}
    >
      <IconClose class="w-4 h-4" />
    </button>
  </div>

  {#if descriptionConfigured}
    {#if description != null && description.trim().length > 0}
      <p class="mt-1.5 text-sm text-slate-600 dark:text-slate-300 whitespace-pre-wrap break-words">{description}</p>
    {:else}
      <p class="mt-1.5 text-sm italic text-slate-400 dark:text-slate-500">No description</p>
    {/if}
  {/if}

  <div class="mt-2 {gridClass}">
    <div class="col-span-full text-xs font-medium uppercase text-slate-400 dark:text-slate-500">Similar features</div>

    {#if related.status === "loading"}
      <div class="col-span-full py-1.5">
        <Spinner status="Finding similar features…" />
      </div>
    {:else if related.status === "error"}
      <div class="col-span-full py-1.5 text-sm text-red-500">Could not compute similar features.</div>
    {:else if related.items.length === 0}
      <div class="col-span-full py-1.5 text-sm text-slate-400 dark:text-slate-500">No similar features found.</div>
    {:else}
      {#each related.items as item (item.feature)}
        {@render renderRow(item)}
      {/each}
    {/if}
  </div>
</div>
