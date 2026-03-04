// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { Coordinator, wasmConnector } from "@uwdata/mosaic-core";
import * as React from "react";
import { useEffect, useState } from "react";

import { EmbeddingAtlas, registerRenderer } from "embedding-atlas/react";
import { createSampleDataTable } from "../sample_datasets.js";

export default function Component() {
  let [coordinator, _] = useState(() => new Coordinator());
  let [ready, setReady] = useState(false);

  class TagsRenderer {
    node: HTMLElement;

    constructor(node: HTMLElement, props: { value: string }) {
      this.node = node;
      this.update(props);
    }

    update(props: { value: string }) {
      let el = document.createElement("span");
      el.innerText = props.value.toString();
      el.style = "border: 1px solid #ccc; border-radius: 2px; padding: 2px 4px";
      this.node.replaceChildren(el);
    }
  }

  registerRenderer({ name: "custom_renderer", label: "Custom Renderer", renderer: TagsRenderer });

  useEffect(() => {
    async function initialize() {
      const wasm = await wasmConnector();
      coordinator.databaseConnector(wasm);
      await createSampleDataTable(coordinator, "dataset", 100000);
      setReady(true);
    }
    initialize();
  }, []);

  if (ready) {
    return (
      <div className="w-full h-full">
        <EmbeddingAtlas
          coordinator={coordinator}
          data={{
            table: "dataset",
            id: "id",
            text: "text",
            projection: { x: "x", y: "y" },
          }}
          initialState={{
            columnStyles: {
              var_many_category: { renderer: "custom_renderer" },
            },
          }}
        />
      </div>
    );
  } else {
    return <p>Initializing dataset...</p>;
  }
}
