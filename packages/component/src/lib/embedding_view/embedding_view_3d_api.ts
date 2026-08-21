// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { createClassComponent } from "svelte/legacy";

import Component from "./EmbeddingView3D.svelte";

export interface EmbeddingView3DProps {
  /** The data. */
  data: {
    x: Float32Array<ArrayBuffer> | number[];
    y: Float32Array<ArrayBuffer> | number[];
    z: Float32Array<ArrayBuffer> | number[];
    category?: Uint8Array<ArrayBuffer> | number[] | null;
  };
  categoryColors?: string[] | null;
  width?: number;
  height?: number;
  pointSize?: number | null;
  selectedIndices?: number[] | null;
  onPointClick?: ((index: number | null) => void) | null;
  onPointHover?: ((index: number | null) => void) | null;
  downsampleMaxPoints?: number | null;
  activeFilterHidden?: boolean;
  showStatusBar?: boolean;
  onRangeSelect?: ((indices: number[]) => void) | null;
}

export class EmbeddingView3D {
  private component: any;
  private currentProps: EmbeddingView3DProps;

  constructor(target: HTMLElement, props: EmbeddingView3DProps) {
    this.currentProps = { ...props };
    this.component = createClassComponent({ component: Component, target: target, props: props });
  }

  update(props: Partial<EmbeddingView3DProps>) {
    let updates: Partial<EmbeddingView3DProps> = {};
    for (let key in props) {
      if ((props as any)[key] !== (this.currentProps as any)[key]) {
        (updates as any)[key] = (props as any)[key];
        (this.currentProps as any)[key] = (props as any)[key];
      }
    }
    this.component.$set(updates);
  }

  /** Reset the camera to the default fit computed when the dataset first loaded. */
  resetCamera() {
    this.component.resetCamera?.();
  }

  destroy() {
    this.component.$destroy();
  }
}
