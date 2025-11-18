// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { safeJSONStringify } from "./json.js";
import { renderMarkdown } from "./markdown.js";

type ResolvedContent = { type: "text"; text: string } | { type: "image"; imageUrl: string };

interface ResolvedMessage {
  role: string;
  content: ResolvedContent[];
  remaining: any;
}

function resolveContent(value: any, output: ResolvedContent[]) {
  if (value == null) {
    return [];
  }
  if (typeof value == "string" && value.length > 0) {
    output.push({ type: "text", text: value });
  } else if (value instanceof Array) {
    for (let item of value) {
      resolveContent(item, output);
    }
  } else if (typeof value == "object") {
    if (value.text != undefined && typeof value.text == "string" && value.text.length > 0) {
      output.push({ type: "text", text: value.text });
    }
    if (value.image != undefined && typeof value.image == "string" && value.image.length > 0) {
      output.push({ type: "image", imageUrl: value.image });
    }
    if (value.image_url != undefined && typeof value.image_url == "string" && value.image_url.length > 0) {
      output.push({ type: "image", imageUrl: value.image_url });
    }
  }
}

function resolveMessage(item: any): ResolvedMessage | undefined {
  if (item == null || typeof item != "object") {
    return;
  }

  let role = item.role?.toString() ?? "(null)";
  let content: ResolvedContent[] = [];
  let remaining = { ...item };

  delete remaining["role"];

  for (let key of ["content", "contents"]) {
    let value = item[key];
    if (value != null) {
      resolveContent(value, content);
      delete remaining[key];
    }
  }

  for (let key in remaining) {
    if (remaining[key] == null) {
      delete remaining[key];
    }
  }

  return { role, content, remaining };
}

export class MessagesRenderer {
  element: HTMLDivElement;

  constructor(element: HTMLDivElement, props: { value: any }) {
    this.element = element;
    this.update(props);
  }

  update(props: { value: any }) {
    let div = document.createElement("div");
    if (props.value == null) {
      div.innerText = "(null)";
    } else if (typeof props.value == "string") {
      div.innerText = props.value;
    } else if (props.value instanceof Array) {
      for (let item of props.value) {
        let resolved = resolveMessage(item);
        if (resolved == undefined) {
          continue;
        }
        div.appendChild(
          E("div", {
            class: "mb-1 flex flex-col gap-1",
            children: [
              // Role
              E("div", {
                class:
                  "text-xs font-bold border-b text-gray-400 dark:text-gray-500 border-gray-400 dark:border-gray-500",
                innerText: resolved.role,
              }),
              // Content
              ...resolved.content.map((c) => {
                if (c.type == "text") {
                  return E("div", {
                    class: "prose dark:prose-invert max-w-none",
                    innerHTML: renderMarkdown(c.text),
                  });
                } else if (c.type == "image") {
                  return E("img", {
                    class: "max-w-120 max-h-120 object-contain",
                    attrs: {
                      src: c.imageUrl,
                    },
                  });
                }
              }),

              // Remaining Properties
              Object.keys(resolved.remaining).length > 0
                ? E("pre", {
                    class:
                      "border rounded-md p-1 bg-gray-100 border-gray-200 dark:bg-gray-800 dark:border-gray-700 text-xs",
                    innerText: safeJSONStringify(resolved.remaining, 2),
                  })
                : null,
            ],
          }),
        );
      }
    }
    this.element.replaceChildren(div);
  }
}

function E(
  tag: string,
  options: {
    innerText?: string;
    innerHTML?: string;
    class?: string;
    attrs?: Record<string, any>;
    children?: (HTMLElement | null | undefined)[];
  },
) {
  let e = document.createElement(tag);
  if (options.innerText != null) {
    e.innerText = options.innerText;
  }
  if (options.innerHTML != null) {
    e.innerHTML = options.innerHTML;
  }
  if (options.class != null) {
    e.className = options.class;
  }
  if (options.attrs != null) {
    for (let [key, value] of Object.entries(options.attrs)) {
      e.setAttribute(key, value);
    }
  }
  if (options.children != null) {
    e.replaceChildren(...options.children.filter((x) => x != null));
  }
  return e;
}
