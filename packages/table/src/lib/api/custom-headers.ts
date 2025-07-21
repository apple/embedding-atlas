// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import type { Component } from "svelte";
import { createClassComponent } from "svelte/legacy";

export type AdditionalHeaderContentsConfig = { [col: string]: AdditionalHeaderContent };

export interface AdditionalHeaderContentProps {
  column: string;
}

type AdditionalHeaderContentClass = new (
  node: HTMLElement,
  props: AdditionalHeaderContentProps,
) => { update?: (props: AdditionalHeaderContentProps) => void; destroy?: () => void };

export type AdditionalHeaderContent =
  | {
      class: AdditionalHeaderContentClass;
      props?: AdditionalHeaderContentProps;
    }
  | AdditionalHeaderContentClass;

export function createAdditionalHeaderContentClass<Props extends AdditionalHeaderContentProps>(
  Component: Component<Props>,
): any {
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
