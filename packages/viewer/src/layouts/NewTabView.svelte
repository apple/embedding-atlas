<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import Button from "../widgets/Button.svelte";
  import CheckBox from "../widgets/CheckBox.svelte";
  import SegmentedControl from "../widgets/SegmentedControl.svelte";

  import { IconDashboardLayout, IconListLayout } from "../assets/icons.js";

  import { getStoreContext } from "../stores/embedding_atlas_store.js";
  interface Props {
    onCreate?: () => void;
  }

  let { onCreate }: Props = $props();

  const store = getStoreContext();

  let populateDefaultCharts = $state.raw(false);
  let layoutType = $state.raw<"list" | "dashboard">("dashboard");

  async function create() {
    if (populateDefaultCharts) {
      await store.addLayoutWithDefaultCharts(layoutType);
    } else {
      store.addLayout(layoutType);
    }
    onCreate?.();
  }
</script>

<div class="flex flex-col gap-2">
  <div class="flex items-center gap-2">
    <span class="text-sm text-slate-500 dark:text-slate-400 w-16 select-none">Layout</span>
    <SegmentedControl
      value={layoutType}
      options={[
        {
          value: "list",
          icon: IconListLayout,
          title:
            "Default layout places the embedding and instance views in separate panels, and other charts in a list / grid",
          label: "Default",
        },
        {
          value: "dashboard",
          icon: IconDashboardLayout,
          title: "Freeform layout allows you to place charts freely",
          label: "Freeform",
        },
      ]}
      onChange={(v) => (layoutType = v as "list" | "dashboard")}
    />
  </div>
  <div class="flex items-center gap-2">
    <span class="text-sm text-slate-500 dark:text-slate-400 w-16 select-none">Charts</span>
    <CheckBox bind:checked={populateDefaultCharts} label="Populate with default charts" />
  </div>
  <div class="flex items-center gap-2">
    <span class="text-sm text-slate-500 dark:text-slate-400 w-16"></span>
    <Button onClick={create} label="Create Tab" class="justify-center" />
  </div>
</div>
