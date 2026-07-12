// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { describe, expect, it, vi } from "vitest";

import { initializeRendererWithFallback, RENDERER_UNAVAILABLE_MESSAGE } from "../src/lib/renderer_selection.js";

describe("initializeRendererWithFallback", () => {
  it("uses WebGPU without trying WebGL2 when WebGPU succeeds", async () => {
    const tryWebGPU = vi.fn().mockResolvedValue(true);
    const tryWebGL2 = vi.fn().mockReturnValue(true);

    await expect(initializeRendererWithFallback(tryWebGPU, tryWebGL2)).resolves.toBe("webgpu");
    expect(tryWebGL2).not.toHaveBeenCalled();
  });

  it("falls back to WebGL2 when WebGPU is unavailable", async () => {
    const tryWebGPU = vi.fn().mockResolvedValue(false);
    const tryWebGL2 = vi.fn().mockReturnValue(true);

    await expect(initializeRendererWithFallback(tryWebGPU, tryWebGL2)).resolves.toBe("webgl2");
    expect(tryWebGL2).toHaveBeenCalledOnce();
  });

  it("returns null when neither renderer can be initialized", async () => {
    await expect(
      initializeRendererWithFallback(
        async () => false,
        () => false,
      ),
    ).resolves.toBeNull();
  });

  it("still tries WebGL2 when WebGPU initialization throws", async () => {
    const error = new Error("WebGPU failed");
    const onError = vi.fn();

    await expect(
      initializeRendererWithFallback(
        async () => {
          throw error;
        },
        () => true,
        onError,
      ),
    ).resolves.toBe("webgl2");
    expect(onError).toHaveBeenCalledWith(error);
  });

  it("reports WebGL2 initialization errors as total failure", async () => {
    const error = new Error("WebGL2 failed");
    const onError = vi.fn();

    await expect(
      initializeRendererWithFallback(
        async () => false,
        () => {
          throw error;
        },
        onError,
      ),
    ).resolves.toBeNull();
    expect(onError).toHaveBeenCalledWith(error);
  });
});

describe("renderer unavailable message", () => {
  it("names both supported rendering APIs and suggests hardware acceleration", () => {
    expect(RENDERER_UNAVAILABLE_MESSAGE).toContain("WebGPU");
    expect(RENDERER_UNAVAILABLE_MESSAGE).toContain("WebGL 2");
    expect(RENDERER_UNAVAILABLE_MESSAGE).toContain("hardware acceleration");
  });
});
