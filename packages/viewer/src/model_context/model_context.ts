// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { validate } from "json-schema";
import { get } from "svelte/store";
import * as z from "zod";

import { delay } from "@embedding-atlas/utils";
import type { MCPTool, ModelContextAPI, ToolResponse } from "../app/mcp_server.js";
import { renderersList } from "../renderers/renderer_types.js";
import {
  schemaBuiltinChartSpec,
  schemaBuiltinChartState,
  schemaBuiltinLayoutSpec,
  schemaColumnStyle,
} from "../schemas.js";
import type { EmbeddingAtlasStore } from "../stores/embedding_atlas_store.js";
import { screenshot, type ScreenshotOptions } from "../utils/screenshot.js";

export interface ModelContextDelegate {
  container: HTMLDivElement;
}

interface ToolDef<S extends z.core.$ZodShape = {}> {
  args?: S;
  description: string;
  handler: (args: z.core.infer<z.ZodObject<S>>) => Promise<unknown>;
}

export class EmbeddingAtlasControl {
  private store: EmbeddingAtlasStore;
  private delegate: ModelContextDelegate;
  private tools: MCPTool[] = [];
  private screenshotOptions: ScreenshotOptions = { maxWidth: 1568, maxHeight: 1568, pixelRatio: 2 };

  constructor(store: EmbeddingAtlasStore, delegate: ModelContextDelegate) {
    this.store = store;
    this.delegate = delegate;
    this.defineTools();
  }

  mcpTools(): MCPTool[] {
    return this.tools;
  }

  private register<S extends z.core.$ZodShape>(name: string, def: ToolDef<S>): void {
    let argsSchema = z.object(def.args ?? ({} as S));
    this.tools.push({
      name,
      description: def.description,
      inputSchema: z.toJSONSchema(argsSchema) as any,
      execute: async (input) => {
        input = argsSchema.parse(input);
        let result = await def.handler(input);
        return resultToToolResponse(result);
      },
    });
  }

  private defineTools() {
    let store = this.store;

    // Data

    this.register("data_schema", {
      description: "Get data schema including name of the primary table and columns",
      handler: async () => ({
        table: store.props.data.table,
        columns: get(store.columns),
      }),
    });

    this.register("data_query", {
      args: {
        query: z.string().describe("The SQL query to run. Must keep this readonly - the server does not enforce it."),
      },
      description: "Run a readonly SQL query in DuckDB",
      handler: async ({ query }) => {
        // TODO: enforce readonly query.
        let result = await store.coordinator.query(query);
        return result.toArray();
      },
    });

    // Get the full app state

    this.register("app_state_get", {
      description: "Get the full app state, including all charts, layouts, and column styles",
      handler: async () => get(store.state),
    });

    // Column Style

    this.register("column_style_list", {
      description: unindent(`
        Get a list of supported column styles (aka., renderers) for values in the table, instance cards, or tooltip.
        Note that column styles only affect the default instance view card.
        To customize cards further, you need to use a custom cardTemplate in the instance view.
      `),
      handler: async () => renderersList,
    });

    this.register("column_style_get", {
      description: "Get column styles for all columns",
      handler: async () => get(store.columnStyles),
    });

    this.register("column_style_schema", {
      description: "Get the column style schema in JSON schema format",
      handler: async () => schemaColumnStyle,
    });

    this.register("column_style_set", {
      args: {
        column: z.string(),
        style: z
          .any()
          .describe(
            "the column style, use column_style_schema to understand the schema; set to {} to reset to default style",
          ),
      },
      description: "Set column style for a given column",
      handler: async ({ column, style }) => {
        store.setColumnStyle(column, style);
        return { column, style };
      },
    });

    // Chart and Layout Tabs

    this.register("chart_spec_schema", {
      description: "Get the chart spec schema in JSON schema format",
      handler: async () => schemaBuiltinChartSpec,
    });

    this.register("chart_state_schema", {
      description: "Get the chart state schema in JSON schema format",
      handler: async () => schemaBuiltinChartState,
    });

    this.register("chart_list", {
      args: {
        includeSpecs: z.boolean().default(false).describe("include the full specs, default false"),
      },
      description: "List all charts in all layout tabs",
      handler: async ({ includeSpecs }) => {
        if (includeSpecs) {
          return get(store.charts);
        } else {
          return Object.fromEntries(
            Object.entries(get(store.charts)).map(([key, spec]) => [
              key,
              {
                title: spec.title,
                type: spec.type,
              },
            ]),
          );
        }
      },
    });

    this.register("chart_get_spec", {
      args: { id: z.string() },
      description: "Get the spec of a chart",
      handler: async (args) => get(store.charts)[args.id],
    });

    this.register("chart_get_state", {
      args: { id: z.string() },
      description: "Get the state of a chart",
      handler: async (args) => get(store.chartStates)[args.id] ?? {},
    });

    this.register("chart_set_spec", {
      args: { id: z.string(), spec: z.any() },
      description: "Set the spec of a chart",
      handler: async ({ id, spec }) => {
        let validateResult = validate(spec, schemaBuiltinChartSpec);
        if (validateResult.valid) {
          store.updateChart(id, spec);
          return { id, spec };
        } else {
          return { error: "invalid chart spec", details: validateResult.errors };
        }
      },
    });

    this.register("chart_set_state", {
      args: { id: z.string(), state: z.any() },
      description: "Set the state of a chart",
      handler: async ({ id, state }) => {
        store.updateChartState(id, state ?? {});
        return { id, state };
      },
    });

    this.register("chart_clear_state", {
      args: { id: z.string() },
      description: "Clear the state of a chart (aka., reset to default)",
      handler: async ({ id }) => {
        store.updateChartState(id, {});
        return { id, state: {} };
      },
    });

    this.register("chart_create", {
      args: {
        layoutId: z.string().describe("The layout id to add the chart to"),
        spec: z.any().describe("The chart spec, use chart_spec_schema to understand the schema"),
      },
      description: "Create a new chart in the specified layout",
      handler: async ({ layoutId, spec }) => {
        let validateResult = validate(spec, schemaBuiltinChartSpec);
        if (validateResult.valid) {
          let id = store.addChartToLayout(layoutId, spec);
          return { id, spec };
        } else {
          return { error: "invalid chart spec", details: validateResult.errors };
        }
      },
    });

    this.register("chart_remove", {
      args: {
        layoutId: z.string().describe("The layout id to remove the chart from"),
        chartId: z.string().describe("The chart id to remove"),
      },
      description: "Remove a chart from the specified layout",
      handler: async ({ layoutId, chartId }) => {
        store.removeChartFromLayout(layoutId, chartId);
        return { chartId, layoutId };
      },
    });

    this.register("layout_spec_schema", {
      description: "Get the layout spec schema in JSON schema format",
      handler: async () => schemaBuiltinLayoutSpec,
    });

    this.register("layout_list", {
      description: "List all layouts (aka., tabs)",
      handler: async () => get(store.layouts),
    });

    this.register("layout_get_current", {
      description: "Get the id of the current layout",
      handler: async () => get(store.currentLayout),
    });

    this.register("layout_set_current", {
      args: { id: z.string() },
      description: "Set the id of the current layout",
      handler: async ({ id }) => {
        store.setCurrentLayout(id);
        return { currentLayout: get(store.currentLayout) };
      },
    });

    this.register("layout_get", {
      args: { id: z.string() },
      description: "Get the spec of a layout",
      handler: async ({ id }) => get(store.layouts)[id],
    });

    this.register("layout_set", {
      args: { id: z.string(), spec: z.any() },
      description: "Set the spec of a layout",
      handler: async ({ id, spec }) => {
        let validateResult = validate(spec, schemaBuiltinLayoutSpec);
        if (validateResult.valid) {
          store.updateLayout(id, spec);
          return { id, spec };
        } else {
          return { error: "invalid layout spec", details: validateResult.errors };
        }
      },
    });

    this.register("layout_set_order", {
      args: { order: z.array(z.string()).describe("The ordered list of layout ids") },
      description: "Set the order of layouts (tabs)",
      handler: async ({ order }) => {
        store.setLayoutOrder(order);
        return { layoutOrder: get(store.layoutOrder) };
      },
    });

    this.register("layout_create", {
      args: { type: z.literal(["list", "dashboard"]) },
      description: "Create a new empty layout",
      handler: async ({ type }) => {
        let id = store.addLayout(type);
        return { id };
      },
    });

    this.register("layout_remove", {
      args: { id: z.string().describe("The layout id to remove") },
      description: "Remove a layout (aka., tab)",
      handler: async ({ id }) => {
        store.removeLayout(id);
        return { id };
      },
    });

    this.register("chart_screenshot_get", {
      args: { id: z.string() },
      description: "Get a screenshot of a chart",
      handler: async (args) => {
        await delay(200);
        let items = this.store.chartDelegates.get(args.id);
        if (items != null) {
          for (let chart of items) {
            if (chart.screenshot) {
              return new ImageResponse(await chart.screenshot(this.screenshotOptions));
            }
          }
        }
        return "chart does not support taking screenshot";
      },
    });

    this.register("app_screenshot_get", {
      description: "Get a full screenshot of the application",
      handler: async () => {
        await delay(200);
        return new ImageResponse(await screenshot(this.delegate.container, this.screenshotOptions));
      },
    });

    this.register("best_practices", {
      description: "Get best practices on how to use tools from Embedding Atlas. Read this before making changes.",
      handler: async () => {
        return unindent(`
          ## Charts and Dashboard

          - Ideally charts should respond to the global cross filter, use "$filter" to refer to the cross filter.
            When applicable, it'd be nice if charts can show unfiltered as a backdrop.

          - If applicable, try adding selections to charts so user can brush the data.

          - The system have a nice scale inference logic that automatically picks scale domain, range, and also scale type
            based on data distribution. If not strong reason leave the scales to default.

          - There might be a very large amount of data, do not make charts that show every individual point,
            unless specifically required, or if we know the number of points is small.

          - When building a dashboard, try to fit all content into a single 24x18 view.
            Having to scroll makes it hard to use the dashboard.

          - It's often useful to have an instance view to see what's being selected.
            If you decide to show cards, it's often useful to make a nice card template.

          - For the markdown widget, use the "title" field in the spec for the title. Start with a lower-level heading, like h2.

          - Do not share charts between layouts.

          - After making big changes, it's useful to take a screenshot of the app and review it.
        `);
      },
    });
  }
}

function unindent(str: string): string {
  let lines = str.split("\n");
  while (lines.length > 0 && lines[0].trim() === "") lines.shift();
  while (lines.length > 0 && lines[lines.length - 1].trim() === "") lines.pop();
  let minIndent = Math.min(...lines.filter((l) => l.trim() !== "").map((l) => l.match(/^ */)![0].length));
  return lines.map((l) => l.slice(minIndent)).join("\n");
}

export function provideModelContext(api: ModelContextAPI, store: EmbeddingAtlasStore, delegate: ModelContextDelegate) {
  let control = new EmbeddingAtlasControl(store, delegate);
  api.provideContext({ tools: control.mcpTools() });
}

class ImageResponse {
  constructor(public dataUrl: string) {}
}

function resultToToolResponse(result: unknown): ToolResponse {
  if (result instanceof ImageResponse) {
    let parsed = parseImageDataUrl(result.dataUrl);
    if (parsed) {
      return { content: [{ type: "image", data: parsed.data, mimeType: parsed.mimeType }] };
    }
    return { content: [{ type: "text", text: "failed to take screenshot" }] };
  }
  if (result === undefined) {
    return { content: [{ type: "text", text: "undefined" }] };
  }
  if (typeof result === "string") {
    return { content: [{ type: "text", text: result }] };
  }
  return { content: [{ type: "text", text: JSON.stringify(result) }] };
}

function parseImageDataUrl(dataUrl: string): { mimeType: string; data: string } | null {
  if (!dataUrl.startsWith("data:")) {
    return null;
  }

  const commaIndex = dataUrl.indexOf(",");
  if (commaIndex === -1) {
    return null;
  }

  const metadata = dataUrl.substring(5, commaIndex);
  const base64Content = dataUrl.substring(commaIndex + 1);

  let mimeType: string;

  if (metadata.includes(";base64")) {
    mimeType = metadata.replace(";base64", "");
  } else if (metadata.includes(";")) {
    mimeType = metadata.split(";")[0];
  } else {
    mimeType = metadata;
  }

  if (!mimeType.startsWith("image/")) {
    return null;
  }

  if (mimeType !== "image/png" && mimeType !== "image/jpeg") {
    return null;
  }

  return { mimeType, data: base64Content };
}
