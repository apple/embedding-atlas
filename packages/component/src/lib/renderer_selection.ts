// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

export type RendererBackend = "webgpu" | "webgl2";

export const RENDERER_UNAVAILABLE_TITLE = "Unable to render embedding";

export const RENDERER_UNAVAILABLE_MESSAGE =
  "Embedding Atlas requires WebGPU or WebGL 2, but neither could be initialized. Check that browser hardware acceleration is enabled and that your browser and graphics drivers are up to date.";

/** Try WebGPU first, then WebGL2, without letting one failed attempt abort fallback. */
export async function initializeRendererWithFallback(
  tryWebGPU: () => boolean | Promise<boolean>,
  tryWebGL2: () => boolean | Promise<boolean>,
  onError: (error: unknown) => void = console.error,
): Promise<RendererBackend | null> {
  try {
    if (await tryWebGPU()) {
      return "webgpu";
    }
  } catch (error) {
    onError(error);
  }

  try {
    if (await tryWebGL2()) {
      return "webgl2";
    }
  } catch (error) {
    onError(error);
  }

  return null;
}
