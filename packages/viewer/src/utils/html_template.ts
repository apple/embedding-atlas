// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { Liquid } from "liquidjs";
import { marked } from "marked";

const engine = new Liquid();

import { sanitizeHTML } from "./sanitize.js";

/** Render markdown into sanitized HTML, safe for innerHTML */
export function renderMarkdown(content: string): string {
  let html = marked(content, { async: false, gfm: true });
  return sanitizeHTML(html);
}

/** Compile a Liquid template into a function that takes value and returns sanitized HTML, safe for innerHTML */
export function compileLiquidTemplate(template: string): (value: any) => string {
  try {
    let parsed = engine.parse(template);
    return (value) => {
      try {
        return sanitizeHTML(engine.renderSync(parsed, value));
      } catch (_) {
        return "Error in Liquid template.";
      }
    };
  } catch (_) {
    return () => "Error in Liquid template.";
  }
}
