// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { coordinator, type Coordinator } from "@uwdata/mosaic-core";
import { getContext, setContext } from "svelte";

const COORDINATOR_KEY = Symbol("mosaic-coordinator");

export class CoordinatorContext {
  public static get coordinator(): Coordinator {
    return getContext(COORDINATOR_KEY) ?? coordinator();
  }

  public static set coordinator(coordinator: Coordinator | null) {
    setContext(COORDINATOR_KEY, coordinator);
  }
}
