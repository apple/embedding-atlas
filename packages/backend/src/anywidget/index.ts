// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { EmbeddingAtlas, type EmbeddingAtlasProps, type EmbeddingAtlasState } from "@embedding-atlas/viewer";
import { type Connector, Coordinator, decodeIPC } from "@uwdata/mosaic-core";

import type { AnyModel, Initialize, Render } from "anywidget/types";

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
          const table = decodeIPC(buffers[0].buffer as any);
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
  const coordinator = new Coordinator();
  coordinator.databaseConnector(makeDatabaseConnector(view.model));

  function saveState(state: EmbeddingAtlasState) {
    view.model.set("_state", JSON.stringify(state));
    view.model.set("_predicate", state.predicate ?? null);
    view.model.save_changes();
  }

  function getProps(): EmbeddingAtlasProps {
    let props = view.model.get("_props");
    let stateJSON = view.model.get("_state");
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
      onStateChange: debounce(saveState, 200),
    };
  }

  // Configure view style
  let container = document.createElement("div");
  container.style.height = "650px";
  container.style.resize = "vertical";
  container.style.overflow = "auto";
  view.el.replaceChildren(container);

  // Create the component
  const component = new EmbeddingAtlas(container, getProps());

  return () => {
    component.destroy();
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

export default { initialize, render };
