<script setup>
import { data } from "./examples.data";
</script>

# Examples

> **Note:** Opening an example will load the dataset directly into your browser from the external provider indicated. Dataset sources and references are listed at the bottom of this page.

Explore these interactive examples showcasing how Embedding Atlas visualizes and analyzes various datasets with pre-computed embeddings.

<div class="example-grid">
  <ExampleItem
    v-for="example of data.examples.embedding"
    :title="example.title"
    :details="example.details"
    :image="example.image"
    :data="example.data"
    :settings="example.settings"
    :state="example.state"
  />
</div>

While Embedding Atlas is primarily designed for visualizing and exploring embeddings, it also provides powerful capabilities for analyzing and visualizing tabular datasets.

<div class="example-grid">
  <ExampleItem
    v-for="example of data.examples.tabular"
    :title="example.title"
    :details="example.details"
    :image="example.image"
    :data="example.data"
    :settings="example.settings"
    :state="example.state"
  />
</div>

### Dataset References

<ul>
  <li v-for="dataset of data.datasets.sort((a, b) => a.title.toUpperCase() < b.title.toUpperCase() ? -1 : 1)" style="line-height: 1.2em; margin: 1em 0;">
      <b>{{dataset.title}}</b>
      <br />
      <span style="font-size: 13px">{{dataset.authors}}</span>
      <br />
      <span style="font-size: 13px">
        <a :href="dataset.link.url" target="_blank" noreferrer noopener>{{dataset.link.title}}</a>
      </span>
  </li>
</ul>

<style scoped>
.example-grid {
  display: flex;
  flex-direction: row;
  flex-wrap: wrap;
  align-items: start;
  gap: 16px;
  margin-top: 16px;
  margin-bottom: 16px;
}
</style>
