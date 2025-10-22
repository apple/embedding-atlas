<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  interface Props {
    extensions?: string[];
    onUpload: (file: File) => void;
  }

  let { extensions, onUpload }: Props = $props();

  let isDragging = $state(false);
  let fileInput: HTMLInputElement;

  function handleDragOver(event: any) {
    event.preventDefault();
    isDragging = true;
  }

  function handleDragLeave() {
    isDragging = false;
  }

  function handleDrop(event: any) {
    event.preventDefault();
    isDragging = false;
    if (event.dataTransfer.files && event.dataTransfer.files.length == 1) {
      let file = event.dataTransfer.files[0];
      if (isValidFile(file)) {
        onUpload(file);
      }
    }
  }

  function handleFileSelect(event: any) {
    const selectedFiles: File[] = Array.from(event.target.files);
    if (selectedFiles.length == 1) {
      if (isValidFile(selectedFiles[0])) {
        onUpload(selectedFiles[0]);
      }
    }
  }

  function isValidFile(file: File) {
    if (extensions == null) {
      return true;
    }
    for (let ext of extensions) {
      if (file.name.toLowerCase().endsWith(ext.toLowerCase())) {
        return true;
      }
    }
    return false;
  }
</script>

<!-- svelte-ignore a11y_no_static_element_interactions -->
<button
  class="flex flex-col items-center w-[40rem] justify-center py-20 border-2 border-dashed rounded-md transition-all
    {isDragging
    ? 'bg-slate-50 border-slate-500 dark:bg-slate-900 dark:border-slate-500'
    : 'bg-slate-50 border-slate-300 dark:bg-slate-900 dark:border-slate-700'}"
  ondragover={handleDragOver}
  ondragleave={handleDragLeave}
  ondrop={handleDrop}
  onclick={() => {
    fileInput.click();
  }}
>
  <div class="text-center space-y-2">
    <p class="text-slate-500 dark:text-slate-500">
      {isDragging ? "Drop your files here" : "Drag & drop files here or click to select"}
    </p>
    <p class="text-sm text-slate-400 dark:text-slate-600">Accepted file types: JSON, CSV, Parquet</p>
  </div>
  <input bind:this={fileInput} type="file" class="hidden" accept={extensions?.join(",")} onchange={handleFileSelect} />
</button>
