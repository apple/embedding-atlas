// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import DefaultTheme from "vitepress/theme";
import "./custom.css";

import ExampleItem from "./ExampleItem.vue";

export default {
  extends: DefaultTheme,
  enhanceApp(ctx) {
    ctx.app.component("ExampleItem", ExampleItem);
  },
};
