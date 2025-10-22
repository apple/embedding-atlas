// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { imageToDataUrl } from "../utils/image.js";

export class ImageRenderer {
  element: HTMLDivElement;

  constructor(element: HTMLDivElement, props: { value: any }) {
    this.element = element;
    this.update(props);
  }

  update(props: { value: any; size?: number }) {
    if (props.value == null) {
      this.element.innerText = "(null)";
      return;
    }
    let dataUrl = imageToDataUrl(props.value);
    if (dataUrl != null) {
      let size = props.size ?? 100;
      let img = document.createElement("img");
      img.src = dataUrl;
      img.style.maxHeight = size + "px";
      img.style.maxWidth = size + "px";
      this.element.replaceChildren(img);
    } else {
      this.element.innerText = `(unknown)`;
    }
  }
}
