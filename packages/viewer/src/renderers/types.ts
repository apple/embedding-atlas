// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

export type CustomComponentClass<N, P> = new (
  node: N,
  props: P,
) => { update?: (props: P) => void; destroy?: () => void };

export interface RendererProps {
  value: any;
  options?: Record<string, any>;
}

export interface RendererOptionsProps {
  options?: Record<string, any>;
  onChange?: (value?: Record<string, any>) => void;
}

/** Component for a custom value renderer */
export type RendererComponent = CustomComponentClass<HTMLElement, RendererProps>;

/** Component for a custom value renderer's options config panel */
export type RendererOptionsComponent = CustomComponentClass<HTMLElement, RendererOptionsProps>;

/** A type describing how to display a column in the table, tooltip, and search results */
export interface ColumnStyle {
  /**
   * The renderer name. Builtin options:
   * - "markdown": Render the value as Markdown
   * - "liquid-template": Render the value with a Liquid template (rendered with liquidjs). Options: template (string): the template, default to "{{ value }}".
   * - "image": Render an image. Options: size (number): the max width/height of the image.
   * - "url": Render the value as a link
   * - "json": Render the value as a JSON string
   * - "messages": Render chat messages (OpenAI format)
   */
  renderer?: string;

  /** Options passed to the renderer class as props */
  options?: Record<string, any>;

  /** Display style in the tooltip */
  display?: "full" | "badge" | "hidden";
}
