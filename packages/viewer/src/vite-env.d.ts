// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

/// <reference types="svelte" />
/// <reference types="vite/client" />

declare module "*&json-schema" {
  import type { JSONSchema7 } from "json-schema";
  const schema: JSONSchema7;
  export default schema;
}
