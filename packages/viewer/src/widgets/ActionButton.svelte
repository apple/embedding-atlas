<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import SvgSpinner from "~icons/svg-spinners/270-ring-with-bg";

  import Button from "./Button.svelte";

  import { IconCheck, IconError } from "../assets/icons.js";

  interface Props {
    label?: string | null;
    icon?: any | null;
    title?: string;
    order?: number | null;
    class?: string | null;
    onClick?: () => Promise<void>;
  }

  let { label = null, icon = null, title = "", order = null, onClick, class: additionalClasses }: Props = $props();

  let state: "ready" | "success" | "running" | "error" = $state("ready");

  let timerClearSuccess: any | null = null;

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
      state = "success";
      if (timerClearSuccess != null) {
        clearTimeout(timerClearSuccess);
      }
      timerClearSuccess = setTimeout(() => {
        if (state == "success") {
          state = "ready";
        }
      }, 2000);
    } catch (e) {
      state = "error";
      console.error(e);
    }
  }
</script>

<Button
  label={label}
  icon={state == "ready" ? icon : state == "running" ? SvgSpinner : state == "success" ? IconCheck : IconError}
  title={title}
  order={order}
  class={additionalClasses}
  onClick={onClickButton}
/>
