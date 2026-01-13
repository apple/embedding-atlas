<script lang="ts">
  interface Props {
    bars: {
      value: number;
      color: string;
    }[];

    maxValue: number;
    width: number;

    label?: string;
    title?: string;

    selected: boolean;
  }

  let props: Props = $props();
</script>

{#if props.width > 10}
  <div class="flex-1 h-4 relative overflow-hidden">
    {#each props.bars as bar}
      {@const w = Math.max((bar.value / props.maxValue) * props.width, bar.value > 0 ? 0.25 : 0)}
      <div
        class="absolute left-0 top-0 bottom-0 rounded-sm"
        style:background={bar.color}
        style:width="{Math.min(w, props.width)}px"
      ></div>
    {/each}
    {#if !props.bars.every((x) => x.value <= props.maxValue)}
      {#each { length: 3 } as _, i}
        <div
          class="absolute top-0 bottom-0 bg-white dark:bg-black w-[1.5px]"
          style:left="{props.width - i * 3 - 4}px"
        ></div>
      {/each}
    {/if}
  </div>
{:else}
  <div class="flex-1"></div>
{/if}
<div class="flex-none">
  <span
    class="text-slate-400 dark:text-slate-500"
    class:!text-gray-200={!props.selected}
    class:dark:!text-gray-600={!props.selected}
    title={props.title}
  >
    {props.label}
  </span>
</div>
