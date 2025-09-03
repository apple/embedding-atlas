// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

export function safeJSONStringify(value: any, space?: number): string {
  try {
    return JSON.stringify(
      value,
      (_, value) => {
        if (value instanceof Object && ArrayBuffer.isView(value)) {
          return Array.from(value as any);
        }
        return value;
      },
      space,
    );
  } catch (e) {
    return "(invalid)";
  }
}

export class JSONRenderer {
  element: HTMLDivElement;

  constructor(element: HTMLDivElement, props: { value: any }) {
    this.element = element;
    this.update(props);
  }

  update(props: { value: any }) {
    let pre = document.createElement("pre");
    pre.style.whiteSpace = "pre-wrap";
    pre.style.wordBreak = "break-all";
    pre.innerText = safeJSONStringify(props.value, 2);
    this.element.replaceChildren(pre);
  }
}
