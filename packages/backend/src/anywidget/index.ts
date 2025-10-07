// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { EmbeddingAtlas, type EmbeddingAtlasProps, type EmbeddingAtlasState } from "@embedding-atlas/viewer";
import { type Connector, Coordinator, decodeIPC } from "@uwdata/mosaic-core";
import type { AnyModel, Initialize, Render } from "anywidget/types";

import { parseCSSColor } from "../utils.js";

interface Model {
  _props: Omit<EmbeddingAtlasProps, "coordinator">;
  _state: string | null;
  _predicate: string | null;
}

function makeDatabaseConnector(model: AnyModel<Model>): Connector {
  const openQueries = new Map();

  function send(query: any, resolve: (value: any) => void, reject: (reason?: any) => void) {
    const uuid = "id-" + Date.now() + "-" + Math.random().toString(36).substr(2, 12);
    openQueries.set(uuid, { query, resolve, reject });
    model.send({ ...query, uuid });
  }

  model.on("msg:custom", (msg, buffers) => {
    const query = openQueries.get(msg.uuid);
    openQueries.delete(msg.uuid);
    if (msg.error) {
      query.reject(msg.error);
    } else {
      switch (msg.type) {
        case "arrow": {
          const table = decodeIPC(buffers[0].buffer);
          query.resolve(table);
          break;
        }
        case "json": {
          query.resolve(msg.result);
          break;
        }
        default: {
          query.resolve({});
          break;
        }
      }
    }
  });

  return {
    query(query: any): Promise<any> {
      return new Promise((resolve, reject) => send(query, resolve, reject));
    },
  };
}

const initialize: Initialize<Model> = (view) => {};

const render: Render<Model> = (view) => {
  // Configure view style
  let container = document.createElement("div");
  container.className = "embedding-atlas-widget-container";
  container.style.height = "650px";
  container.style.resize = "vertical";
  container.style.overflow = "hidden";

  view.el.replaceChildren(container);

  if (isVSCode()) {
    // Stop wheel event propagation to prevent scrolling problems.
    container.onwheel = (e) => {
      e.stopPropagation();
    };

    // Override the ".cell-output-ipywidget-background" class to remove padding and white background.
    let style = document.createElement("style");
    style.textContent = `
      .cell-output-ipywidget-background:has(.embedding-atlas-widget-container) {
        background: transparent !important;
        padding-left: 0 !important;
        padding-right: 0 !important;
      }
    `;
    view.el.appendChild(style);
  }

  const coordinator = new Coordinator();
  coordinator.databaseConnector(makeDatabaseConnector(view.model));

  let component: EmbeddingAtlas | null = null;

  function saveState(state: EmbeddingAtlasState) {
    view.model.set("_state", JSON.stringify(state));
    view.model.set("_predicate", state.predicate ?? null);
    view.model.save_changes();
  }

  function getProps(): EmbeddingAtlasProps | null {
    let props = view.model.get("_props");
    let stateJSON = view.model.get("_state");
    if (props == null || props.data == null) {
      // The props is not ready yet
      return null;
    }
    let state: EmbeddingAtlasState | null = null;
    try {
      if (stateJSON) {
        state = JSON.parse(stateJSON);
      }
    } catch (e) {}

    return {
      coordinator: coordinator,
      ...props,
      ...(state != null ? { initialState: state } : {}),
      colorScheme: detectColorScheme(container),
      onStateChange: debounce(saveState, 200),
    };
  }

  function createOrUpdate() {
    let props = getProps();
    if (props == null) {
      return;
    }
    if (component != null) {
      component.update(props);
    } else {
      container.replaceChildren();
      component = new EmbeddingAtlas(container, props);

      let currentColorScheme = props.colorScheme;
      let observer = new MutationObserver(() => {
        let newValue = detectColorScheme(container);
        if (newValue !== currentColorScheme && newValue != null) {
          component?.update({ colorScheme: newValue });
          currentColorScheme = newValue;
        }
      });
      observer.observe(document.body, { attributes: true, attributeFilter: ["style", "class"] });
    }
  }

  view.model.on("change:_props", () => {
    createOrUpdate();
  });

  setTimeout(createOrUpdate, 10);

  return () => {
    component?.destroy();
    coordinator.clear();
  };
};

function debounce<T extends any[]>(func: (...args: T) => void, time: number = 1000): (...args: T) => void {
  let timeout: any | undefined = undefined;
  let perform = (...args: T) => {
    if (timeout) {
      clearTimeout(timeout);
    }
    timeout = setTimeout(() => {
      func(...args);
    }, time);
  };
  return perform;
}

function isVSCode() {
  return typeof (window as any).vscIPyWidgets !== "undefined";
}

function detectColorScheme(container: HTMLElement): "light" | "dark" | undefined {
  let color: string | undefined = undefined;
  if (isVSCode()) {
    color = getComputedStyle(container).getPropertyValue("--theme-background")?.trim();
  } else {
    color = getComputedStyle(container).getPropertyValue("--jp-layout-color0")?.trim();
  }

  if (color == undefined || color.trim() == "") {
    if (document.body.classList.contains("light")) {
      return "light";
    }
    if (document.body.classList.contains("dark")) {
      return "dark";
    }
    return undefined;
  }
  let { r, g, b } = parseCSSColor(color);
  let grayscale = 0.299 * r + 0.587 * g + 0.114 * b;
  return grayscale < 128 ? "dark" : "light";
}

export default { initialize, render };
