# EmbeddingViewMosaic

The `embedding-atlas` package contains a component for displaying up to a few millions of points from an embedding with x and y coordinates.
The component connects to a [Mosaic](https://idl.uw.edu/mosaic/) coordinator and can display data for specified table and x, y coordinate columns.

We also provide React and Svelte wrappers of the component to easily include it in your own application.

<p class="light-only"><img style="margin: 0 auto;" src="./public/assets/component-light.png" /></p>
<p class="dark-only"><img style="margin: 0 auto;" src="./public/assets/component-dark.png" /></p>

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
