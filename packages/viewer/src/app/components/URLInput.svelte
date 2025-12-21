<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import Button from "../../widgets/Button.svelte";
  import Input from "../../widgets/Input.svelte";

  import { IconImport } from "../../assets/icons.js";

  interface Props {
    onConfirm: (url: string) => void;
  }

  let { onConfirm }: Props = $props();

  let inputValue = $state("");

  function confirm() {
    let url = inputValue.trim();
    if (url == "") {
      return;
    }
    onConfirm(url);
  }
</script>

<div class="flex gap-1">
  <Input className="flex-1" bind:value={inputValue} placeholder="Enter data URL" onEnter={confirm} />
  <Button onClick={confirm} icon={IconImport} />
</div>
