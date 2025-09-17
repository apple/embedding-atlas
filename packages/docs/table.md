# Table

The `embedding-atlas` package contains a table component for showing data frames from Mosaic.

```bash
npm install embedding-atlas
```

To use the React wrapper:

```js
import { Table } from "embedding-atlas/react";

<Table
  table="data_table"
  columns={['id', 'column1', 'column2']}
  rowKey="id"
  ...
/>
```

To use the Svelte wrapper:

```js
import { Table } from "embedding-atlas/svelte";

<Table
  table="data_table"
  columns={['id', 'column1', 'column2']}
  rowKey="id"
  ...
/>
```

## API

The Table component accepts a number of requiremed and optional properties.

<!-- @doc(ts): TableProps -->

## Custom Cells

To use custom cell rendering, first create a class for the custom cell renderer:

```ts
interface CustomCellProps {
  value: any;
  rowData: any;
}

class CustomCellRenderer {
  constructor(target, props: CustomCellProps) {
    // Create the cell component and mount it to the target element.
  }
  update(props: CustomCellProps) {
    // Update the component with new props.
  }
  destroy() {
    // Destroy the component.
  }
}
```

Then specify the `customCells` property to the component for the desired column:

```svelte
<EmbeddingViewMosaic
  ...
  customCells={{
    columnA: CustomCellRenderer,
  }}
/>
```

## Additional Header Contents

Similar to custom cells, to add additional header content, first create a class for the additional header content:

```ts
interface AdditionalHeaderContentProps {
  column: string;
}

class AdditionalHeaderContentRenderer {
  constructor(target, props: AdditionalHeaderContentProps) {
    // Create the cell component and mount it to the target element.
  }
  update(props: AdditionalHeaderContentProps) {
    // Update the component with new props.
  }
  destroy() {
    // Destroy the component.
  }
}
```

Then specify the `additionalHeaderContents` property to the component for the desired column:

```svelte
<EmbeddingViewMosaic
  ...
  additionalHeaderContents={{
    columnA: AdditionalHeaderContentRenderer,
  }}
/>
```
