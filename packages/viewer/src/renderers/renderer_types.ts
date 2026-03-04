// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import type { Component } from "svelte";
import type { Action } from "svelte/action";
import { createClassComponent } from "svelte/legacy";

import ImageOptions from "./ImageOptions.svelte";
import LiquidTemplateOptions from "./LiquidTemplateOptions.svelte";

import { compileLiquidTemplate, renderMarkdown } from "../utils/html_template.js";
import { imageToDataUrl } from "../utils/image.js";
import { renderMessages } from "./messages.js";
import { safeJSONStringify } from "./renderer_utils.js";
import type {
  CustomComponentClass,
  RendererComponent,
  RendererOptionsComponent,
  RendererOptionsProps,
  RendererProps,
} from "./types.js";

export let renderers: Record<string, Action<HTMLElement, RendererProps>> = {};
export let rendererOptions: Record<string, Action<HTMLElement, RendererOptionsProps>> = {};
export let renderersList: { renderer: string; label: string; description?: string }[] = [];

export function registerRenderer(options: {
  name: string;
  label?: string;
  description?: string;
  renderer: RendererComponent;
  options?: RendererOptionsComponent;
}) {
  renderers[options.name] = classToAction(options.renderer);
  if (options.options) {
    rendererOptions[options.name] = classToAction(options.options);
  } else {
    delete rendererOptions[options.name];
  }

  let desc = {
    renderer: options.name,
    label: options.label ?? options.name,
    description: options.description,
  };
  let idx = renderersList.findIndex((r) => r.renderer == desc.renderer);
  if (idx < 0) {
    renderersList.push(desc);
  } else {
    renderersList[idx] = desc;
  }
}

function classToAction<E, T>(Component: CustomComponentClass<E, T>) {
  return (node: E, props: T) => {
    let instance = new Component(node, props);
    return {
      update: instance.update?.bind(instance),
      destroy: instance.destroy?.bind(instance),
    };
  };
}

function registerSimpleRenderer(options: {
  name: string;
  label?: string;
  description?: string;
  renderer: (node: HTMLElement, props: RendererProps) => void;
  options?: Component<RendererOptionsProps>;
}) {
  renderersList.push({
    renderer: options.name,
    label: options.label ?? options.name,
    description: options.description,
  });
  renderers[options.name] = (node, props) => {
    options.renderer(node, props);
    return {
      update: (props) => {
        options.renderer(node, props);
      },
    };
  };
  if (options.options) {
    let Class = options.options;
    rendererOptions[options.name] = (node, props) => {
      let c: any = createClassComponent({ component: Class, target: node, props: props });
      return {
        update: (props) => {
          c?.$set(props);
        },
        destroy: () => {
          c.$destroy();
          c = null;
        },
      };
    };
  }
}

// Builtin renderers
registerSimpleRenderer({
  name: "markdown",
  label: "Markdown",
  description: "Render the value as Markdown",
  renderer: (node, props) => {
    node.innerHTML = `<div class="markdown-content">` + renderMarkdown(props.value?.toString() ?? "(null)") + `</div>`;
  },
});

registerSimpleRenderer({
  name: "liquid-template",
  label: "Liquid Template",
  description: "Render the value with a Liquid template (with liquidjs)",
  renderer: (node, props) => {
    node.innerHTML =
      `<div>` + compileLiquidTemplate(props.options?.template ?? "{{value}}")({ value: props.value }) + `</div>`;
  },
  options: LiquidTemplateOptions,
});

registerSimpleRenderer({
  name: "image",
  label: "Image",
  description: "Render the value as an image. Expect image data.",
  renderer: (node, props) => {
    if (props.value == null) {
      node.innerText = "(null)";
      return;
    }
    let dataUrl = imageToDataUrl(props.value);
    if (dataUrl != null) {
      let size = props.options?.size ?? 100;
      let img = document.createElement("img");
      img.referrerPolicy = "no-referrer";
      img.src = dataUrl;
      img.style.maxHeight = size + "px";
      img.style.maxWidth = size + "px";
      node.replaceChildren(img);
    } else {
      node.innerText = `(unknown)`;
    }
  },
  options: ImageOptions,
});

registerSimpleRenderer({
  name: "url",
  label: "Link",
  description: "Render the value as a link. Expect a URL.",
  renderer: (node, props) => {
    if (props.value != null) {
      let a = document.createElement("a");
      a.href = props.value;
      a.innerText = props.value;
      a.className = "underline";
      a.target = "_blank";
      a.rel = "noopener noreferrer";
      node.replaceChildren(a);
    } else {
      node.replaceChildren();
      node.innerText = `(null)`;
    }
  },
});

registerSimpleRenderer({
  name: "json",
  label: "JSON",
  description: "Render the value as JSON",
  renderer: (node, props) => {
    let pre = document.createElement("pre");
    pre.className = "text-sm";
    pre.style.whiteSpace = "pre-wrap";
    pre.style.wordBreak = "break-all";
    pre.innerText = safeJSONStringify(props.value, 2);
    node.replaceChildren(pre);
  },
});

registerSimpleRenderer({
  name: "messages",
  label: "Messages",
  description: "Render the value as chat messages (OpenAI format)",
  renderer: renderMessages,
});
