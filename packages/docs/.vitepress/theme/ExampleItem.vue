<script setup>
import { withBase } from "vitepress";
import { computed } from "vue";

const props = defineProps(["title", "details", "image", "data", "settings", "state"]);
const href = computed(
  () =>
    "/examples/app/#?" +
    new URLSearchParams({
      data: props.data,
      ...(props.settings != null ? { settings: props.settings } : {}),
      ...(props.state != null ? { state: props.state } : {}),
    }).toString(),
);
</script>

<template>
  <a :href="withBase(href)" target="_blank" class="example-item">
    <img :src="withBase(image)" />
    <div class="example-item-title">{{ props.title }}</div>
    <div class="example-item-details">{{ props.details }}</div>
  </a>
</template>

<style scoped>
.example-item {
  display: flex;
  flex-direction: column;
  gap: 4px;
  width: 336px;
  padding: 8px;
  line-height: 1em;
  font-size: 14px;
  border-radius: 8px;
  color: var(--vp-c-indigo-1);
  border: 1px solid var(--vp-c-gray-1);
  background: var(--vp-c-gray-soft);
  text-decoration: none !important;
  transition: all 0.3s ease;
  cursor: pointer;
}

.example-item:hover {
  color: var(--vp-c-indigo-1);
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
  border-color: var(--vp-c-brand-1);
}

.example-item img {
  border-radius: 2px;
  aspect-ratio: 16 / 9;
  object-fit: cover;
}

.example-item-title {
  margin-top: 2px;
}

.example-item-details {
  font-size: 10px;
  color: #999;
}
</style>
