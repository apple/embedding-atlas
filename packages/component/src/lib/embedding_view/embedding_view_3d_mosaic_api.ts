// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import type { Coordinator, Selection } from "@uwdata/mosaic-core";
import { createClassComponent } from "svelte/legacy";

import Component from "./EmbeddingView3DMosaic.svelte";

import type { DataField, DataPoint } from "./types.js";

export interface EmbeddingView3DMosaicProps {
  coordinator?: Coordinator;
  table: string;
  x: string;
  y: string;
  z: string;
  category?: string | null;
  identifier?: string | null;
  additionalFields?: Record<string, DataField> | null;
  filter?: Selection | null;
  categoryColors?: string[] | null;
  width?: number | null;
  height?: number | null;
  pointSize?: number | null;
  selection?: DataPoint[] | null;
  onSelection?: ((value: DataPoint[] | null) => void) | null;
  onTooltip?: ((value: DataPoint | null) => void) | null;
  downsampleMaxPoints?: number | null;
  showStatusBar?: boolean;
}

export class EmbeddingView3DMosaic {
  private component: any;
  private currentProps: EmbeddingView3DMosaicProps;

  constructor(target: HTMLElement, props: EmbeddingView3DMosaicProps) {
    this.currentProps = { ...props };
    this.component = createClassComponent({ component: Component, target: target, props: props });
  }

  update(props: Partial<EmbeddingView3DMosaicProps>) {
    let updates: Partial<EmbeddingView3DMosaicProps> = {};
    for (let key in props) {
      if ((props as any)[key] !== (this.currentProps as any)[key]) {
        (updates as any)[key] = (props as any)[key];
        (this.currentProps as any)[key] = (props as any)[key];
      }
    }
    this.component.$set(updates);
  }

  destroy() {
    this.component.$destroy();
  }
}
