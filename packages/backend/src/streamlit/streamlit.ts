// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { debounce } from "@embedding-atlas/utils";
import { EmbeddingAtlas, type EmbeddingAtlasProps } from "@embedding-atlas/viewer";
import { Coordinator, wasmConnector } from "@uwdata/mosaic-core";
import { ArrowTable, type RenderData, Streamlit } from "streamlit-component-lib";

const coordinator = new Coordinator();
const connector = wasmConnector();
coordinator.databaseConnector(connector);

// The root container element
const container = document.createElement("div");
container.style.width = "100%";
container.style.height = "800px";
container.style.borderRadius = "4px";
container.style.overflow = "hidden";
container.style.resize = "vertical";
document.body.appendChild(container);

const resizeObserver = new ResizeObserver(() => {
  Streamlit.setFrameHeight();
});
resizeObserver.observe(container);

// The embedding atlas view instance
let view: EmbeddingAtlas | null = null;

// The current component value, used to stop setting it if it hasn't changed.
let currentComponentValue: any = null;

let debouncedSetValue = debounce((value) => {
  if (JSON.stringify(currentComponentValue) != JSON.stringify(value)) {
    currentComponentValue = value;
    Streamlit.setComponentValue(value);
  }
}, 300);

async function createView(data_frame: ArrowTable, props: Partial<EmbeddingAtlasProps>) {
  if (view) {
    view.destroy();
    container.replaceChildren();
  }
  let conn = await connector.getConnection();
  const ipcBuffer = data_frame.serialize().data;
  await conn.insertArrowFromIPCStream(ipcBuffer, { name: "dataframe" });
  const row_id_column = "__row_index__";
  await coordinator.exec(`
    CREATE SEQUENCE row_id_sequence MINVALUE 0 START 0;
    ALTER TABLE dataframe ADD COLUMN IF NOT EXISTS ${row_id_column} INTEGER DEFAULT nextval('row_id_sequence');
  `);
  view = new EmbeddingAtlas(container, {
    ...props,
    coordinator: coordinator,
    data: {
      ...props.data,
      table: "dataframe",
      id: row_id_column,
    },
    onStateChange: (state) => {
      debouncedSetValue({
        predicate: state.predicate,
      });
    },
  });
}

let needsToCreateView = true;

function onRender(event: Event): void {
  const data = (event as CustomEvent<RenderData>).detail;
  const data_frame = data.args.data_frame;
  const props: Partial<EmbeddingAtlasProps> = {
    ...data.args.props,
    colorScheme: data.theme?.base ?? "light",
  };
  if (needsToCreateView) {
    needsToCreateView = false;
    createView(data_frame, props);
  } else {
    view?.update(props);
  }
  Streamlit.setFrameHeight();
}

Streamlit.events.addEventListener(Streamlit.RENDER_EVENT, onRender);
Streamlit.setComponentReady();
Streamlit.setFrameHeight();
