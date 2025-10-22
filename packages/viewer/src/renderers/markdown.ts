// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { marked } from "marked";

export class MarkdownRenderer {
  element: HTMLDivElement;

  constructor(element: HTMLDivElement, props: { value: any }) {
    this.element = element;
    this.update(props);
  }

  update(props: { value: any }) {
    this.element.innerHTML =
      `<div class="markdown-content">` +
      marked(props.value?.toString() ?? "(null)", { async: false, gfm: true }) +
      `</div>`;
  }
}
