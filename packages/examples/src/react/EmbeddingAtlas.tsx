// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { Coordinator, wasmConnector } from "@uwdata/mosaic-core";
import * as React from "react";
import { useEffect, useState } from "react";

import { EmbeddingAtlas } from "embedding-atlas/react";
import { createSampleDataTable } from "../sample_datasets";

export default function Component() {
  let [coordinator, _] = useState(() => new Coordinator());
  let [ready, setReady] = useState(false);

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
        />
      </div>
    );
  } else {
    return <p>Initializing dataset...</p>;
  }
}
