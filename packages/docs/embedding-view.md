# EmbeddingView

The `embedding-atlas` package contains a component for displaying up to a few millions of points from an embedding with x and y coordinates.

We also provide React and Svelte wrappers of the component to easily include it in your own application.

<p class="light-only"><img style="margin: 0 auto;" src="./assets/component-light.png" /></p>
<p class="dark-only"><img style="margin: 0 auto;" src="./assets/component-dark.png" /></p>

```bash
npm install embedding-atlas
```

To use the React wrapper:

```js
import { EmbeddingView } from "embedding-atlas/react";

// Capture the view's tooltip with a state
let [tooltip, setTooltip] = useState(null);

let xColumn: Float32Array;
let yColumn: Float32Array;
let categoryColumn: Uint8Array; // optional

<EmbeddingView
  data={{x: xColumn, y: yColumn, category: categoryColumn}}
  tooltip={tooltip}
  onTooltip={setTooltip}
  ...
/>
```

To use the Svelte wrapper:

```js
import { EmbeddingView } from "embedding-atlas/svelte";

// The tooltip as a state.
let tooltip = $state(null);

<EmbeddingView
  data={{ x: xColumn, y: yColumn, category: categoryColumn }}
  tooltip={tooltip}
  onTooltip={(v) => (tooltip = v)}
  ...
/>
```

If your application does not use React or Svelte, you may directly construct the component:

```js
import { EmbeddingView } from "embedding-atlas";

let target = document.getElementById("container");
let props = {
  data: {
    x: xColumn,
    y: yColumn,
    category: categoryColumn,
    onTooltip: (value) => {
      // ...
    },
  },
};

// Create and mount the component
let component = new EmbeddingView(target, props);

// Update with new props
component.update(newProps);

// Destroy the component
component.destroy();
```

## Properties

The view can be configured with the following properties (props):

<!-- @doc(ts): EmbeddingViewProps -->

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
<EmbeddingView
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
<EmbeddingView
  ...
  customOverlay={{
    class: CustomOverlay,
    props: { customProp: 10 } // Pass additional props to the overlay component.
  }}
/>
```
