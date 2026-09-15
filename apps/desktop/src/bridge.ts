// Electron bridge — mimics the @tauri-apps/api surface the Svelte app
// used to import, so App.svelte's call sites stay unchanged in shape.

interface BridgeEvent<T> {
  payload: T;
}

interface ElectronBridge {
  invoke<T = unknown>(cmd: string, args?: unknown): Promise<T>;
  listen<T = unknown>(event: string, cb: (ev: BridgeEvent<T>) => void): () => void;
  openDialog(options: OpenOptions): Promise<string | string[] | null>;
  reportDroppedFile(path: string): Promise<void>;
  getPathForFile(file: File): string;
}

interface OpenOptions {
  multiple?: boolean;
  directory?: boolean;
  filters?: Array<{ name: string; extensions: string[] }>;
}

declare global {
  interface Window {
    electronAPI: ElectronBridge;
  }
}

function hasBridge(): boolean {
  return typeof window !== "undefined" && !!window.electronAPI;
}

export function invoke<T = unknown>(cmd: string, args?: Record<string, unknown>): Promise<T> {
  if (!hasBridge()) {
    return Promise.reject(
      new Error(
        `electronAPI unavailable — invoke("${cmd}") called from a context without the preload (likely a non-Electron browser)`,
      ),
    );
  }
  return window.electronAPI.invoke<T>(cmd, args);
}

// Tauri's listen returns Promise<UnlistenFn>; ours is synchronous but we
// wrap it so call sites that do `.then(f => f())` still work. In a
// non-Electron browser, the listener is a no-op — keeps the page loadable
// in plain Chrome during dev without immediately crashing.
export function listen<T = unknown>(event: string, cb: (ev: BridgeEvent<T>) => void): Promise<() => void> {
  if (!hasBridge()) return Promise.resolve(() => {});
  return Promise.resolve(window.electronAPI.listen<T>(event, cb));
}

export async function open(options: OpenOptions): Promise<string | null> {
  if (!hasBridge()) return null;
  const result = await window.electronAPI.openDialog(options);
  if (result == null) return null;
  if (Array.isArray(result)) return result[0] ?? null;
  return result;
}
