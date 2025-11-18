// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import DOMPurify from "dompurify";
import { marked } from "marked";

export function renderMarkdown(content: string): string {
  let html = marked(content, { async: false, gfm: true });
  return DOMPurify.sanitize(html);
}

export class MarkdownRenderer {
  element: HTMLDivElement;

  constructor(element: HTMLDivElement, props: { value: any }) {
    this.element = element;
    this.update(props);
  }

  update(props: { value: any }) {
    this.element.innerHTML =
      `<div class="markdown-content">` + renderMarkdown(props.value?.toString() ?? "(null)") + `</div>`;
  }
}
