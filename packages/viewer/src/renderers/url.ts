// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

export class URLRenderer {
  element: HTMLDivElement;

  constructor(element: HTMLDivElement, props: { value: any }) {
    this.element = element;
    this.update(props);
  }

  update(props: { value: any }) {
    if (props.value != null) {
      let a = document.createElement("a");
      a.href = props.value;
      a.innerText = props.value;
      a.className = "underline";
      a.target = "_blank";
      this.element.replaceChildren(a);
    } else {
      this.element.innerText = `(null)`;
    }
  }
}
