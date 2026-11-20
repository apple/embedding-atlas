# EmbeddingViewMosaic

The `embedding-atlas` package contains a component for displaying up to a few million points from an embedding with x and y coordinates.
The component connects to a [Mosaic](https://idl.uw.edu/mosaic/) coordinator and can display data for specified table and x, y coordinate columns.

We also provide React and Svelte wrappers for the component to easily include it in your own application.

<p class="light-only"><img style="margin: 0 auto;" src="./public/assets/component-light.avif" loading="lazy" /></p>
<p class="dark-only"><img style="margin: 0 auto;" src="./public/assets/component-dark.avif" loading="lazy" /></p>

```bash
npm install embedding-atlas
```

To use the React wrapper:

```js
import { EmbeddingViewMosaic } from "embedding-atlas/react";

<EmbeddingViewMosaic
  table="data_table"
  x="x_column"
  y="y_column"
  category="category_column"
  text="text_column"
  identifier="identifier_column"
  filter={brush}
  ...
/>
```

To use the Svelte wrapper:

```js
import { EmbeddingViewMosaic } from "embedding-atlas/svelte";

<EmbeddingViewMosaic
  table="data_table"
  x="x_column"
  y="y_column"
  category="category_column"
  text="text_column"
  identifier="identifier_column"
  filter={brush}
  ...
/>
```

If your application does not use React or Svelte, you may directly construct the component:

```js
import { EmbeddingViewMosaic } from "embedding-atlas";

let target = document.getElementById("container");
let props = {
  table: "data_table",
  x: "x_column",
  y: "y_column",
  category: "category_column",
  text: "text_column",
  identifier: "identifier_column",
  filter: brush,
  onTooltip: (value) => {
    // ...
  },
};

// Create and mount the component
let component = new EmbeddingViewMosaic(target, props);

// Update with new props
component.update(newProps);

// Destroy the component
component.destroy();
```

## Properties

The view can be configured with the following properties (props):

<!-- @doc(ts): EmbeddingViewMosaicProps -->

## Layers

By default, the view draws the data as points (with a density map in the `"density"` mode), and adds labels on top.
Use the `layers` property to control what is drawn. There are three types of layers: `"points"`, `"density"`, and `"labels"`.
Each layer can have its own `data` (table and fields), `filter`, and colors, which default to the view's, so you can overlay multiple tables in the same view:

```js
<EmbeddingViewMosaic
  table="data_table"
  x="x_column"
  y="y_column"
  category="category_column"
  text="text_column"
  filter={brush}
  layers={[
    // Density map and contours of the main table
    { type: "density" },
    // Points of another table, not filtered
    { type: "points", data: { table: "other_table", x: "x", y: "y" }, filter: null, color: "#e31a1c" },
    // Labels generated once from the main table's text column (all data)
    { type: "labels" },
    // Labels that are regenerated when the filter changes
    { type: "labels", filter: brush },
  ]}
  ...
/>
```

If a layer uses the same table as the view (or doesn't specify `data.table`), its fields default to the view's fields.
If a layer uses a different table, `data.x` and `data.y` must be specified.
Points and density layers on the view's table use the view's `filter` by default; layers on other tables are not filtered by default, since the view's `filter` applies to the view's table. Labels layers are not filtered by default, so automatic labels stay stable. Set a labels layer's `filter` to have its labels follow the filter.

Points and density layers are blended together regardless of their order; labels are drawn on top of them, and labels from all labels layers are placed together so they don't overlap.
The tooltip and selection use the view's main data (`table`, `x`, `y`, and `filter`).

### Layer Data

<!-- @doc(ts): LayerData -->

### Points Layer

<!-- @doc(ts): PointsLayer_2 -->

### Density Layer

<!-- @doc(ts): DensityLayer_2 -->

### Labels Layer

<!-- @doc(ts): LabelsLayer_2 -->

## Fields and Colors

The fields (`x`, `y`, `category`, `text`, `image`, and `importance`, also in a layer's `data`) can be column names or SQL expressions, e.g., `{ sql: "x * 2" }`. A layer's expressions are evaluated for the layer's table. The `identifier` must be a column name.

The `category` field colors the points and splits the density. Use `colorScale` to set how its values map to colors:

```js
<EmbeddingViewMosaic
  category="topic"
  colorScale={{ domain: ["sports", "politics", "science"], range: ["#1f77b4", "#ff7f0e", "#2ca02c"] }}
  ...
/>
```

Values are compared as text, so a domain can also list numbers or booleans. Values not in the domain use the `other` color, and null values use the `null` color.
To color numeric values by bins, specify `thresholds` instead of `domain`; values outside the thresholds use the `other` color, and null and NaN values use the `null` color.
Without a domain or thresholds, the category values must be category indices (integers starting from 0), colored by `colorScale.range` (or `categoryColors`).

Without a `category`, the points and density use the fixed `color`.

<!-- @doc(ts): ColorScale -->

## Config

You can pass in an object with the following properties to the `config` property of the embedding view:

<!-- @doc(ts): EmbeddingViewConfig -->

## Theme

You can pass in an object with the following properties to the `theme` property of the embedding view.
You can also provide these options as `light` and/or `dark` properties, which will control the appearance of the view depending on its `colorScheme`. For example:

```ts
{
  light: {
    clusterLabelColor: "black";
  }
  dark: {
    clusterLabelColor: "white";
  }
}
```

<!-- @doc(ts,no-required): EmbeddingViewTheme -->

## Custom Tooltip

You may use the `customTooltip` property to change how tooltips are displayed.

First create a class for the custom tooltip component:

```js
class CustomTooltip {
  constructor(target, props) {
    // Create the tooltip component and mount it to the target element.
    // props will contain a `tooltip` field, plus any custom prop you specified.
  }
  update(props) {
    // Update the component with new props.
  }
  destroy() {
    // Destroy the component.
  }
}
```

Then specify the `customTooltip` property to the component:

```js
<EmbeddingViewMosaic
  ...
  customTooltip={{
    class: CustomTooltip,
    props: { customProp: 10 } // Pass additional props to the tooltip component.
  }}
/>
```

## Custom Overlay

You may use the `customOverlay` property to add an overlay to the embedding view.

First create a class for the custom overlay:

```js
class CustomOverlay {
  constructor(target, props) {
    // Create the tooltip component and mount it to the target element.
    // props will contain a `proxy` field, plus any custom prop you specified.
    // You can use proxy.location(x, y) to get the pixel location of a data point at (x, y).
  }
  update(props) {
    // Update the component with new props.
  }
  destroy() {
    // Destroy the component.
  }
}
```

Then specify the `customOverlay` property to the component:

```js
<EmbeddingViewMosaic
  ...
  customOverlay={{
    class: CustomOverlay,
    props: { customProp: 10 } // Pass additional props to the overlay component.
  }}
/>
```
