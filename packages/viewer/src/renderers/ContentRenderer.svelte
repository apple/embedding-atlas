<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { imageToDataUrl } from "../utils/image.js";
  import { isImage, isLink, stringify, textRendererClasses } from "./index.js";

  interface Props {
    value?: string;
    renderer?: string;
    rendererOptions?: Record<string, any>;
  }

  let { value = "", renderer, rendererOptions = {} }: Props = $props();

  let rendererClass = $derived(renderer != null ? (textRendererClasses[renderer] ?? null) : null);

  function action(element: HTMLDivElement, props: { class: any; value: any; options: Record<string, any> }) {
    let component = new props.class(element, { value: props.value, ...props.options });
    return {
      update(props: { value: any; options: Record<string, any> }) {
        component.update?.({ value: props.value, ...props.options });
      },
      destroy() {
        component.destroy?.();
      },
    };
  }
</script>

{#if rendererClass == null}
  {#if isLink(value)}
    <a href={value} class="underline" target="_blank">{value}</a>
  {:else if isImage(value)}
    <img src={imageToDataUrl(value)} alt="" referrerpolicy="no-referrer" class="max-w-24 max-h-24" />
  {:else}
    {stringify(value)}
  {/if}
{:else}
  {#key rendererClass}
    <div use:action={{ class: rendererClass, value: value, options: rendererOptions ?? {} }}></div>
  {/key}
{/if}
