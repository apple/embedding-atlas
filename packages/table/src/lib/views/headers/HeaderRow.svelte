<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { onMount } from "svelte";

  import Header from "./Header.svelte";
  import Resizer from "./Resizer.svelte";

  import { ConfigContext } from "../../context/config.svelte";
  import { Context } from "../../context/context.svelte";
  import { OID } from "../../model/TableModel.svelte";
  import Dropdown from "../shared/Dropdown.svelte";

  const model = Context.model;
  const controller = Context.controller;
  const config = ConfigContext.config;

  let headerElement: HTMLElement | null = $state(null);
  let scrollContainer: HTMLElement | null = $state(null);
  let dropdownLabelContainer: HTMLElement | null = $state(null);

  let columns: string[] = $derived(model.renderableCols);

  let lastRAFId = 0;

  onMount(() => {
    lastRAFId = requestAnimationFrame(tick);

    return () => {
      cancelAnimationFrame(lastRAFId);
    };
  });

  function tick() {
    if (scrollContainer) {
      scrollContainer.style.transform = `translate3d(${controller.xScroll}px, 0, 0)`;
    }

    if (dropdownLabelContainer) {
      dropdownLabelContainer.style.transform = `translate3d(${-controller.xScroll}px, 0, 0)`;
    }

    lastRAFId = requestAnimationFrame(tick);
  }
</script>

<div class="header-row" bind:this={headerElement}>
  <div class="scroll-container" bind:this={scrollContainer}>
    <div class="dropdown-label-container">
      <div class="dropdown-label" bind:this={dropdownLabelContainer}>
        <Dropdown label="⋮" relativeTo={headerElement}>
          <ul
            class="column-toggle"
            style:--max-height={controller.viewHeight - 48 + "px"}
            style:--max-width={controller.viewWidth - 48 + "px"}
          >
            {#each model.columns as col}
              <li class="column-entry">
                <label class="column-label">
                  {col === OID ? "row #" : (config.columnConfigs[col]?.title ?? col)}
                  <input
                    type="checkbox"
                    id="{col}-checkbox"
                    style:float="right"
                    checked={col === OID ? config.showRowNumber !== false : !config.columnConfigs[col]?.hidden}
                    onchange={(e) => {
                      const target = e.target as HTMLInputElement;
                      const checked = target.checked;
                      if (checked) {
                        controller.showColumn(col);
                      } else {
                        controller.hideColumn(col);
                      }
                    }}
                  />
                </label>
              </li>
            {/each}
          </ul>
        </Dropdown>
      </div>
    </div>
    {#each columns as col (col)}
      <Header col={col} />
      <Resizer col={col} />
    {/each}
  </div>
</div>

<style>
  .header-row {
    flex-shrink: 0;
    border-bottom: 1px solid var(--secondary-bg);
    background-color: var(--primary-bg);
  }

  .scroll-container {
    display: flex;
    flex-direction: row;
  }

  .dropdown-label-container {
    position: absolute;
    z-index: 20;
    left: 0px;
    box-sizing: border-box;
    height: 100%;
    padding: 0.25em;
    display: flex;
    flex-direction: row;
    align-items: end;
  }

  .dropdown-label {
    height: 1.5em;
    align-items: center;
    display: flex;
  }

  .column-toggle {
    margin: 0;
    margin-top: 4px;
    margin-left: 8px;
    padding: 12px;
    background-color: var(--primary-bg);
    border-radius: 4px;
    box-shadow: var(--shadow);
    border: var(--outline);
    max-height: var(--max-height);
    max-width: var(--max-width);
    overflow: scroll;
  }

  .column-entry {
    list-style-type: none;
    padding: 4px;
    user-select: none;
  }

  .column-label {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    color: var(--secondary-text-color);
  }
</style>
