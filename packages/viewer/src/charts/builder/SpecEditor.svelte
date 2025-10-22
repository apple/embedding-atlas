<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import CodeEditor from "../../widgets/CodeEditor.svelte";

  interface Props {
    spec: any;
    colorScheme?: "light" | "dark";
    onSpecChange?: (spec: any) => void;
  }

  let { spec, colorScheme, onSpecChange }: Props = $props();

  let validateResult = $state<{ message?: string; spec?: any } | undefined>(undefined);
  let isValid = $derived(validateResult?.spec !== undefined);

  function validate(textValue: string): typeof validateResult {
    try {
      let newSpec = JSON.parse(textValue);
      return { spec: newSpec };
    } catch (e: any) {
      return { message: e.message?.toString() };
    }
  }

  function confirm() {
    if (validateResult && validateResult.spec != undefined) {
      onSpecChange?.(validateResult.spec);
    }
  }
</script>

<div class="w-full h-full flex flex-col gap-2">
  <CodeEditor
    class="w-full flex-1"
    colorScheme={colorScheme}
    language="json"
    value={JSON.stringify(spec, null, 2)}
    onChange={(newValue) => {
      validateResult = validate(newValue);
    }}
  />
  <div class="flex-none flex gap-2 items-center">
    <button
      class="flex-none px-2 h-8 w-24 rounded-md text-white text-sm"
      class:bg-blue-500={isValid}
      class:bg-gray-300={!isValid}
      class:dark:text-gray-500={!isValid}
      class:dark:bg-gray-700={!isValid}
      disabled={!isValid}
      onclick={confirm}
    >
      Confirm
    </button>
    <div class="flex-1 w-0 overflow-hidden text-nowrap text-ellipsis" title={validateResult?.message ?? ""}>
      {validateResult?.message ?? ""}
    </div>
  </div>
</div>
