<script lang="ts">
  interface Props {
    label: string;
    disabled?: boolean;
    onClick?: () => void;
  }

  let { label, disabled, onClick }: Props = $props();
</script>

<button
  class="px-2 h-8 w-24 rounded-md text-white text-sm select-none {!disabled
    ? 'bg-blue-500'
    : 'bg-gray-300 dark:text-gray-500 dark:bg-gray-700'}"
  disabled={disabled}
  onclick={() => {
    if (!disabled) {
      onClick?.();
    }
  }}
>
  {label}
</button>
