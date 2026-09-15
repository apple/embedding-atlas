// Shared WebGPU feature probe — runs in all three distro forms
// (backend-frontend, frontend-only, and the native macOS app's WKWebView).

export type WebGPUStatus =
  | { kind: "ok"; adapterName: string; device: string }
  | { kind: "missing"; reason: string }
  | { kind: "unsupported"; reason: string };

/** Run a cheap adapter-request probe. Does NOT create a device. */
export async function probeWebGPU(): Promise<WebGPUStatus> {
  const gpu = (navigator as any).gpu;
  if (gpu == null || gpu.requestAdapter == null) {
    return { kind: "missing", reason: "navigator.gpu is not available" };
  }
  if (gpu.wgslLanguageFeatures == null) {
    return { kind: "unsupported", reason: "navigator.gpu.wgslLanguageFeatures missing" };
  }
  if (!gpu.wgslLanguageFeatures.has("unrestricted_pointer_parameters")) {
    return {
      kind: "unsupported",
      reason: "WGSL feature 'unrestricted_pointer_parameters' not supported",
    };
  }
  try {
    const adapter = await gpu.requestAdapter();
    if (adapter == null) {
      return { kind: "unsupported", reason: "no compatible GPU adapter" };
    }
    const info: any = (await adapter.requestAdapterInfo?.()) ?? {};
    const adapterName = info.description || info.device || info.vendor || info.architecture || "";
    return { kind: "ok", adapterName, device: info.device ?? "" };
  } catch (e) {
    return { kind: "unsupported", reason: `requestAdapter failed: ${String(e)}` };
  }
}
