<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts" module>
  import { type Coordinator, isSelection, Param } from "@uwdata/mosaic-core";
  import { astToDOM, parseSpec, type Spec } from "@uwdata/mosaic-spec";
  import { createAPIContext } from "@uwdata/vgplot";
  import { onMount } from "svelte";

  import { Context } from "../contexts.js";
  import { plotColors } from "./colors.js";

  function isSourceSelf(container: HTMLElement, source: any) {
    let plotElement = source.mark?.plot?.element;
    if (plotElement == null) {
      return false;
    }
    return container?.contains(plotElement) ?? false;
  }

  async function makeVgPlotFromSpec(
    coordinator: Coordinator,
    container: HTMLElement,
    spec: Spec | null,
    params: any | null,
  ) {
    if (spec == null || container == null) {
      return null;
    }
    let paramsMap = new Map();
    for (let key in params) {
      paramsMap.set(key, params[key]);
    }
    let ast = parseSpec(spec);
    let apiContext = createAPIContext({ coordinator: coordinator });
    let result = await astToDOM(ast, { params: paramsMap, api: apiContext } as any);
    let element = result.element;
    container.appendChild(element);

    return () => {
      for (let key in params) {
        let selection = params[key];
        if (!isSelection(selection)) {
          continue;
        }
        for (let clause of selection.clauses) {
          if (isSourceSelf(container, clause.source)) {
            clause.source?.reset?.();
            selection.update({ ...clause, value: null, predicate: null });
          }
        }
      }
      element.remove();
    };
  }
</script>

<script lang="ts">
  interface Props {
    spec: Spec | null;
    params: any | null;
  }

  let { spec, params }: Props = $props();
  const coordinator = Context.coordinator;

  let container: HTMLDivElement;

  let containerWidth = $state(100);

  const contextParams = {
    continuousColorScheme: Param.value("YlGnBu"),
    continuousColorSchemeZero: Param.value("black"),
    markColor: Param.value("black"),
    markColorFade: Param.value("black"),
    containerWidth: Param.value(100),
  };

  $effect(() => {
    return Context.darkMode.subscribe((value) => {
      let colors = value ? plotColors.dark : plotColors.light;
      contextParams.continuousColorScheme.update(colors.continuousColorScheme);
      contextParams.continuousColorSchemeZero.update(colors.continuousColorSchemeAtZero);
      contextParams.markColor.update(colors.markColor);
      contextParams.markColorFade.update(colors.markColorFade);
    });
  });

  $effect(() => {
    contextParams.containerWidth.update(containerWidth);
  });

  onMount(() => {
    $effect.pre(() => {
      let destroyFunction: (() => void) | null = null;

      async function makePlot(...args: Parameters<typeof makeVgPlotFromSpec>) {
        destroyFunction = await makeVgPlotFromSpec(...args);
      }

      makePlot(coordinator, container, spec, {
        ...contextParams,
        ...params,
      });

      return () => {
        destroyFunction?.();
      };
    });
  });
</script>

<div class="w-full" bind:this={container} bind:clientWidth={containerWidth}></div>
