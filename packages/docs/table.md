# Table

The `embedding-atlas` package contains a table component for showing data frames from Mosaic.

```
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

### Required

### table `string`

The name of the duckdb table to create a view for.

### columns `string[]`

The columns of the table to render. These should match the names of the columns in the duckdb `table` you wish to render.

### rowKey `string`

The name of the column used to uniquely identify rows.

### Optional

### coordinator? `Coordinator | null`

A Mosaic `Coordinator` for the table, if the default coordinator is not desired.

### columnConfigs? `ColumnConfigs | null`

A `ColumnConfigs` is a `{ [column: string]: ColumnConfig }`, where a `ColumnConfig` has the following options:

- `width?: number` The width of the column, if not provided, a default width will be chosen by the table.
- `title?: string` The string to render as the title of the column.
- `hidden?: boolean` Whether the column should be hidden from the table.

The properties `width` and `hidden` may be modified by the table via UI interactions, with updates provided through the `onColumnConfigsChange` callback.

### onColumnConfigsChange? `((newConfig: ColumnConfigs) => void) | null`

A function that will be called whenever the table changes its `columnConfigs`. You can use this to save configurations through sessions.

### showRowNumber? `boolean`

Whether to show the row number as column.

### onShowRowNumberChange? `((newShowRowNumber: boolean) => void) | null`

A function that will be called whenever the table changes its `showRowNumber` value. You can use this to save configurations through sessions.

### filter? `Selection | null`

A Mosaic `Selection` used to filter the table.

### scrollTo? `any | null`

A rowId to scroll to. When this value is updated, the table will scroll to that row.

### colorScheme? `'light' | 'dark | null'`

Light or dark mode.

### theme? `Theme | null`

A theme object, which has the following options:

- `primaryTextColor?: string` The text color of elements such as cells.
- `secondaryTextColor?: string` The text color of elements such as headers.
- `tertiaryTextColor?: string` The text color of elements such as sort buttons.
- `fontFamily?: string` The font family of text in the table.
- `fontSize?: string` The font size of text in the table.
- `primaryBackgroundColor?: string` The background of elements such as headers and even-numbered cells.
- `secondaryBackgroundColor?: string` The background of elements such as odd-numbered cells.
- `hoverBackgroundColor?: string` The background color of hovered elements such as buttons.
- `headerFontFamily?: string` The font family of the header, will fall back to the table's font family.
- `headerFontSize?: string` The font size of the header, will fall back to the table's font size.
- `cellFontFamily?: string` The font family of the cells, will fall back to the table's font family.
- `cellFontSize?: string` The font size of the cells, will fall back to the table's font size.
- `scrollbarBackgroundColor?: string` The background color of the scrollbars.
- `scrollbarPillColor?: string` The background color of the scrollbar pills.
- `scrollbarLabelBackgroundColor?: string` The background color of the vertical scrollbar label.
- `shadow?: string` The shadow of elements such as overlays.
- `outlineColor?: string` The outline of elements such as overlays.
- `dimmedRowColor?: string` The overlay color for dimmed rows when highlighted rows are present in the table.
- `rowScrollToColor?: string` The color of rows will flash when they're scrolled to using the `scrollTo` property of the table.
- `rowHoverColor?: string` The color of rows when they're hovered, enabled through the `showHoveredRow` property of the table.

These values can be css variables if you wish to use css defined custom properties. For example: `{ primaryTextColor: "var(--my-color-variable)" }`.

You can also provide these options as `light` and `dark` properties, which will control the appearance of the table depending on its `colorScheme`. For example:

```ts
{
  light: {
    primaryTextColor: "black";
  }
  dark: {
    primaryTextColor: "white";
  }
}
```

### lineHeight? `number | null`

The height of each line of text, in pixels. Defaults to `20`.

### numLines? `number | null`

The number of lines of text to show in each row. Defaults to `3`.

### headerHeight? `number | null`

The height of the header, in pixels. Defaults to an auto height based on rendered title.

### onRowClick?: `((rowId: string) => void)| null`

A handler for rows being clicked, which can be used to coordinate with other views.

### highlightedRows: `any[] | null`

When provided, these all other rows will be dimmed in the table.

### customCells? `CustomCellsConfig`

You can use this to designate custom renderers for columns. A `CustomCellsConfig` is a `{ [column: string]: CustomCell }`. See [Custom Cells](#custom-cells) for more info.

### additionalHeaderContents? `AdditionalHeaderContentsConfig`

You can use this to designate additional content for column headers. Additional header content is rendered above the title of the header, and can be used to add helpful content such as summaries and visualizations to headers. A `AdditionalHeaderContentsConfig` is a `{ [column: string]: AdditionalHeaderContent }`. See [Additional Header Contents](#additional-header-contents) for more info.

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
