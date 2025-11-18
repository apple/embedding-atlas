# EmbeddingAtlas

The `embedding-atlas` package contains a component for the entire frontend user interface of Embedding Atlas.

```bash
npm install embedding-atlas
```

To use the React wrapper:

```js
import { EmbeddingAtlas } from "embedding-atlas/react";

let coordinator: Coordinator; // The Mosaic coordinator.

<EmbeddingAtlas
  coordinator={coordinator}
  data={{
    table: "data_table",
    id: "id_column",
    projection: { x: "x_column", y: "y_column" },
    text: "text_column"
  }}
  ...
/>
```

To use the Svelte wrapper:

```js
import { EmbeddingAtlas } from "embedding-atlas/svelte";

let coordinator: Coordinator; // The Mosaic coordinator.

<EmbeddingAtlas
  coordinator={coordinator}
  data={{
    table: "data_table",
    id: "id_column",
    projection: { x: "x_column", y: "y_column" },
    text: "text_column"
  }}
  ...
/>
```

If your application does not use React or Svelte, you may directly construct the component:

```js
import { EmbeddingAtlas } from "embedding-atlas";

let coordinator: Coordinator; // The Mosaic coordinator.

let target = document.getElementById("container");
let props = {
  coordinator: coordinator,
  data: {
    table: "data_table",
    id: "id_column",
    projection: { x: "x_column", y: "y_column" },
    text: "text_column"
  },
  // ...
};

// Create and mount the component
let component = new EmbeddingAtlas(target, props);

// Update with new props
component.update(newProps);

// Destroy the component
component.destroy();
```

## Properties

The view can be configured with the following properties (props):

<!-- @doc(ts): EmbeddingAtlasProps -->

## State

The `EmbeddingAtlasState` interface describes the state of the Embedding Atlas UI.

You may set `initialState` to a previously-saved state value to reload the UI to its previous state.

Properties of the state:

<!-- @doc(ts): EmbeddingAtlasState -->

## Chart Theme

You can pass in an object with the following properties to the `chartTheme` property of the component to style the charts.
You can also provide these options as `light` and/or `dark` properties, which will control the appearance of the view depending on its `colorScheme`. For example:

```ts
{
  light: {
    markColor: "black";
  }
  dark: {
    markColor: "white";
  }
}
```

<!-- @doc(ts,no-required): ChartTheme -->
