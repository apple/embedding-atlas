// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import type { Component } from "svelte";
import { createClassComponent } from "svelte/legacy";

export function createCustomComponentClass<Props extends {}>(Component: Component<Props>): any {
  return class {
    private component: any;

    constructor(target: HTMLDivElement, props: Props) {
      this.component = createClassComponent({ component: Component, target: target, props: props });
    }

    update(props: Partial<Props>) {
      this.component.$set(props);
    }

    destroy() {
      this.component.$destroy();
    }
  };
}
