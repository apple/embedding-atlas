// Electron preload — bridges renderer IPC to main and installs the
// per-page state-sync logic that Tauri injected via eval().
//
// Runs with contextIsolation + sandbox, so the page can only see what we
// expose via contextBridge. We also run DOM-level glue (home button,
// drag overlay, hash persistence) when the page loads — exactly
// replicating what STATE_SYNC_SCRIPT did in lib.rs.

import { contextBridge, ipcRenderer, webUtils, type IpcRendererEvent } from "electron";

// ---------- 1. contextBridge API (the "window.__TAURI__" replacement) ----------

interface BridgeEvent<T> {
  payload: T;
}

const bridge = {
  invoke<T = unknown>(cmd: string, args?: unknown): Promise<T> {
    return ipcRenderer.invoke(`cmd:${cmd}`, args);
  },
  listen<T = unknown>(event: string, cb: (ev: BridgeEvent<T>) => void): () => void {
    const channel = `evt:${event}`;
    const handler = (_e: IpcRendererEvent, payload: T) => cb({ payload });
    ipcRenderer.on(channel, handler);
    return () => {
      ipcRenderer.off(channel, handler);
    };
  },
  openDialog(options: {
    multiple?: boolean;
    directory?: boolean;
    filters?: Array<{ name: string; extensions: string[] }>;
  }): Promise<string | string[] | null> {
    return ipcRenderer.invoke("dialog:open", options);
  },
  reportDroppedFile(path: string): Promise<void> {
    return ipcRenderer.invoke("cmd:dropped_file", { path });
  },
  getPathForFile(file: File): string {
    // Electron's modern replacement for the deprecated `File.path`.
    return webUtils.getPathForFile(file);
  },
};

contextBridge.exposeInMainWorld("electronAPI", bridge);

// ---------- 2. Page-level glue (runs for every URL the BrowserWindow loads) ----------

const HOME_SVG =
  '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" ' +
  'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
  '<path d="M3 9.5L12 3l9 6.5V20a1 1 0 0 1-1 1h-5v-6h-6v6H4a1 1 0 0 1-1-1V9.5z"/></svg>';

function installPageGlue() {
  if ((window as unknown as { __gsaBound?: boolean }).__gsaBound) return;
  (window as unknown as { __gsaBound: boolean }).__gsaBound = true;

  // --- hash persistence (mirror URL-hash changes to the native state file) ---
  let pending: ReturnType<typeof setTimeout> | null = null;
  const save = () => {
    if (pending) clearTimeout(pending);
    pending = setTimeout(() => {
      bridge.invoke("save_viewer_state", { hash: location.hash || "" }).catch(() => {});
    }, 400);
  };
  window.addEventListener("hashchange", save);
  window.addEventListener("popstate", save);
  const origPush = history.pushState;
  history.pushState = function (...args: Parameters<typeof origPush>) {
    origPush.apply(this, args);
    save();
  };
  const origReplace = history.replaceState;
  history.replaceState = function (...args: Parameters<typeof origReplace>) {
    origReplace.apply(this, args);
    save();
  };

  // --- home button (viewer pages only) ---
  const isViewer = location.hostname === "127.0.0.1";
  if (isViewer) {
    const makeBtn = () => {
      const b = document.createElement("button");
      b.id = "gsa-home-btn";
      b.type = "button";
      b.setAttribute("aria-label", "Back to dataset picker");
      b.title = "Back to dataset picker";
      b.innerHTML = HOME_SVG;
      b.style.cssText = [
        "display:inline-flex",
        "align-items:center",
        "justify-content:center",
        "width:32px",
        "height:32px",
        "flex:0 0 auto",
        "border-radius:6px",
        "border:1px solid rgba(100,116,139,0.35)",
        "background:transparent",
        "color:inherit",
        "cursor:pointer",
        "padding:0",
        "margin:0",
      ].join(";");
      b.onmouseenter = () => {
        b.style.background = "rgba(100,116,139,0.18)";
      };
      b.onmouseleave = () => {
        b.style.background = "transparent";
      };
      b.onclick = () => {
        bridge.invoke("return_home").catch(() => {});
      };
      return b;
    };

    const tryInject = (): boolean => {
      const input = document.querySelector('input[type="search"]');
      if (!input) return false;
      const leftSide = input.closest(".flex-1");
      const toolbar = (leftSide && leftSide.parentElement) || input.parentElement;
      if (!toolbar) return false;
      if (toolbar.querySelector("#gsa-home-btn")) return true;
      toolbar.insertBefore(makeBtn(), toolbar.firstChild);
      return true;
    };

    const observer = new MutationObserver(() => {
      tryInject();
    });
    const startObserving = () => {
      if (document.body) {
        observer.observe(document.body, { childList: true, subtree: true });
      } else {
        document.addEventListener("DOMContentLoaded", startObserving, { once: true });
      }
    };
    startObserving();
    setTimeout(tryInject, 100);
    setTimeout(tryInject, 500);
  }

  // --- drag-drop handling ---
  // In Electron the renderer owns the drop: HTML5 events fire, we read the
  // OS path via webUtils.getPathForFile(), and hand it off to main.
  const overlay = document.createElement("div");
  overlay.style.cssText = [
    "position:fixed",
    "inset:0",
    "z-index:2147483646",
    "background:rgba(37,99,235,0.18)",
    "border:3px dashed rgba(96,165,250,0.9)",
    "box-sizing:border-box",
    "display:none",
    "align-items:center",
    "justify-content:center",
    "font:600 18px -apple-system,BlinkMacSystemFont,sans-serif",
    "color:#dbeafe",
    "pointer-events:none",
  ].join(";");
  overlay.textContent = "Drop a dataset to open";
  const attachOverlay = () => {
    if (document.body) document.body.appendChild(overlay);
    else document.addEventListener("DOMContentLoaded", attachOverlay, { once: true });
  };
  attachOverlay();

  window.addEventListener("dragenter", (e) => {
    e.preventDefault();
    overlay.style.display = "flex";
  });
  window.addEventListener("dragover", (e) => {
    e.preventDefault();
    overlay.style.display = "flex";
  });
  window.addEventListener("dragleave", (e) => {
    if (e.clientX <= 0 || e.clientY <= 0 || e.clientX >= window.innerWidth || e.clientY >= window.innerHeight) {
      overlay.style.display = "none";
    }
  });
  window.addEventListener("drop", (e) => {
    e.preventDefault();
    overlay.style.display = "none";
    const files = e.dataTransfer?.files;
    if (!files || files.length === 0) return;
    // Only the first file is used, matching the Tauri behaviour.
    const first = files[0];
    const path = bridge.getPathForFile(first);
    if (path) {
      bridge.reportDroppedFile(path).catch(() => {});
    }
  });
}

// Preload runs before DOMContentLoaded; wait for document to exist.
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", installPageGlue, { once: true });
} else {
  installPageGlue();
}
