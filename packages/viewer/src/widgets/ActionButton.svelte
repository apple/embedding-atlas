<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import SvgSpinner from "~icons/svg-spinners/270-ring-with-bg";

  import Button from "./Button.svelte";

  import { IconError } from "../assets/icons.js";

  interface Props {
    label?: string | null;
    icon?: any | null;
    title?: string;
    order?: number | null;
    class?: string | null;
    onClick?: () => Promise<void>;
  }

  let { label = null, icon = null, title = "", order = null, onClick, class: additionalClasses }: Props = $props();

  let state: "ready" | "running" | "error" = $state("ready");

  async function onClickButton() {
    if (!onClick) {
      return;
    }
    if (state == "running") {
      return;
    }
    state = "running";
    try {
      await onClick();
      state = "ready";
    } catch (e) {
      state = "error";
      console.error(e);
    }
  }
</script>

<Button
  label={label}
  icon={state == "ready" ? icon : state == "running" ? SvgSpinner : IconError}
  title={title}
  order={order}
  class={additionalClasses}
  onClick={onClickButton}
/>
