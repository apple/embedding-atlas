import { validate } from "json-schema";
import type { MCPTool, ModelContextAPI, ToolResponse } from "../app/mcp_server.js";
import type { ChartContext, ChartDelegate } from "../charts/chart.js";
import {
  schemaBuiltinChartSpec,
  schemaBuiltinChartState,
  schemaDashboardLayoutState,
  schemaListLayoutState,
} from "../schemas.js";
import { findUnusedId } from "../utils/identifier.js";
import { screenshot, type ScreenshotOptions } from "../utils/screenshot.js";

export interface ModelContextDelegate {
  context: ChartContext;
  charts: Record<string, any>;
  chartStates: Record<string, any>;
  layout: string;
  layoutStates: Record<string, any>;
  chartDelegates: Map<string, Set<ChartDelegate>>;
  container: HTMLDivElement;
}

export function provideModelContext(api: ModelContextAPI, delegate: ModelContextDelegate) {
  let screenshotOptions: ScreenshotOptions = { maxWidth: 1568, maxHeight: 1568, pixelRatio: 2 };

  let tools: MCPTool[] = [
    {
      name: "get_data_schema",
      description: "Get the table name and columns",
      inputSchema: { type: "object", additionalProperties: false },
      execute: async () => {
        return jsonResponse({
          table: delegate.context.table,
          columns: delegate.context.columns,
        });
      },
    },
    {
      name: "run_sql_query",
      description: "Run a readonly SQL query in DuckDB.",
      inputSchema: {
        type: "object",
        properties: {
          query: {
            type: "string",
            description: `The SQL query to run, must be readonly.`,
          },
        },
        additionalProperties: false,
      },
      execute: async (params: { query: string }) => {
        let result = await delegate.context.coordinator.query(params.query);
        return jsonResponse(result.toArray());
      },
    },
    {
      name: "list_charts",
      description: "List all charts in Embedding Atlas.",
      inputSchema: { type: "object", additionalProperties: false },
      execute: async () => {
        return jsonResponse(delegate.charts);
      },
    },
    {
      name: "add_chart",
      description: "Create a new chart with the specification, returns the id of the new chart.",
      inputSchema: {
        type: "object",
        properties: {
          spec: {
            type: "object",
            description: `
                The chart specification. Schema: ${JSON.stringify(schemaBuiltinChartSpec)}.
                Notes:
                - The data might be very large (>100k) points. Try not to create a chart that has no aggregation.
                - Add "filter": "$filter" to appropriate layers to make the chart respond to filters from other charts. The filter is a cross-filter.
                - When creating a chart, consider adding interactivity to it.
                - The plot size is determined by the chart container by default. Refrain from setting it directly.
                - Before adding a new chart, please list existing charts with list_charts at least once to ensure no duplication.
              `,
          },
        },
        additionalProperties: false,
      },
      execute: async (params: { spec: any }) => {
        // Validate schema.
        let validateResult = validate(params.spec, schemaBuiltinChartSpec);
        if (validateResult.valid) {
          let id = findUnusedId(delegate.charts);
          delegate.charts = { ...delegate.charts, [id]: params.spec };
          return jsonResponse({ id: id });
        } else {
          return jsonResponse({ error: "Spec is invalid", details: validateResult.errors });
        }
      },
    },
    {
      name: "get_chart_spec",
      description: "Get the specification of a chart",
      inputSchema: {
        type: "object",
        properties: {
          id: { type: "string" },
        },
        additionalProperties: false,
      },
      execute: async (params: { id: string; spec: any }) => {
        return jsonResponse(delegate.charts[params.id]);
      },
    },
    {
      name: "set_chart_spec",
      description: "Update the specification of a chart",
      inputSchema: {
        type: "object",
        properties: {
          id: { type: "string" },
          spec: { type: "object", description: "The new chart specification, replacing the existing one." },
        },
        additionalProperties: false,
      },
      execute: async (params: { id: string; spec: any }) => {
        let validateResult = validate(params.spec, schemaBuiltinChartSpec);
        if (validateResult.valid) {
          delegate.charts = { ...delegate.charts, [params.id]: params.spec };
          return textResponse("success");
        } else {
          return jsonResponse({ error: "Spec is invalid", details: validateResult.errors });
        }
      },
    },
    {
      name: "get_chart_state",
      description: "Get the state of a chart",
      inputSchema: {
        type: "object",
        properties: {
          id: { type: "string" },
        },
        additionalProperties: false,
      },
      execute: async (params: { id: string; spec: any }) => {
        return jsonResponse(delegate.chartStates[params.id]);
      },
    },
    {
      name: "set_chart_state",
      description: `
          Update the state of a chart. Schema: ${JSON.stringify(schemaBuiltinChartState)}.
        `,
      inputSchema: {
        type: "object",
        properties: {
          id: { type: "string" },
          state: { type: "object", description: "The new chart state, replacing the existing one." },
        },
        additionalProperties: false,
      },
      execute: async (params: { id: string; state: any }) => {
        delegate.chartStates = { ...delegate.chartStates, [params.id]: params.state };
        return textResponse("success");
      },
    },
    {
      name: "clear_chart_state",
      description: "Clear the state of a chart",
      inputSchema: {
        type: "object",
        properties: {
          id: { type: "string" },
        },
        additionalProperties: false,
      },
      execute: async (params: { id: string; state: any }) => {
        delegate.chartStates = { ...delegate.chartStates, [params.id]: {} };
        return textResponse("success");
      },
    },
    {
      name: "delete_chart",
      description: "Delete a chart",
      inputSchema: {
        type: "object",
        properties: {
          id: { type: "string" },
        },
        additionalProperties: false,
      },
      execute: async (params: { id: string; spec: any }) => {
        delegate.charts = Object.fromEntries(Object.entries(delegate.charts).filter((x) => x[0] != params.id));
        delegate.chartStates = Object.fromEntries(
          Object.entries(delegate.chartStates).filter((x) => x[0] != params.id),
        );
        return textResponse("success");
      },
    },
    {
      name: "get_chart_screenshot",
      description: "Get a screenshot of a chart",
      inputSchema: {
        type: "object",
        properties: {
          id: { type: "string" },
        },
        additionalProperties: false,
      },
      execute: async (params: { id: string }) => {
        let items = delegate.chartDelegates.get(params.id);
        if (items != null) {
          for (let chart of items) {
            if (chart.screenshot) {
              let image = await chart.screenshot(screenshotOptions);
              return imageResponse(image);
            }
          }
        }
        return textResponse("chart does not support taking screenshot");
      },
    },
    {
      name: "get_layout_type",
      description: "Get the type of the current layout ('list' or 'dashboard')",
      inputSchema: {
        type: "object",
        additionalProperties: false,
      },
      execute: async () => {
        return textResponse(delegate.layout);
      },
    },
    {
      name: "set_layout_type",
      description: "Set the type of the current layout  ('list' or 'dashboard')",
      inputSchema: {
        type: "object",
        properties: {
          type: { type: "string" },
        },
        additionalProperties: false,
      },
      execute: async (params: { type: string }) => {
        delegate.layout = params.type;
        return textResponse("success");
      },
    },
    {
      name: "get_layout_state",
      description: "Get the state of the current layout",
      inputSchema: {
        type: "object",
        additionalProperties: false,
      },
      execute: async () => {
        return jsonResponse(delegate.layoutStates[delegate.layout] ?? {});
      },
    },
    {
      name: "set_layout_state",
      description: "Set the state of the current layout",
      inputSchema: {
        type: "object",
        properties: {
          state: {
            type: "object",
            description: `
                The new chart state, replacing the existing one.
                Schema:
                - dashboard layout state: ${JSON.stringify(schemaDashboardLayoutState)}
                - list layout state: ${JSON.stringify(schemaListLayoutState)}
              `,
          },
        },
        additionalProperties: false,
      },
      execute: async (params: { state: any }) => {
        delegate.layoutStates = { ...delegate.layoutStates, [delegate.layout]: params.state };
        return textResponse("success");
      },
    },
    {
      name: "get_full_screenshot",
      description: "Get a full screenshot of the application",
      inputSchema: {
        type: "object",
        additionalProperties: false,
      },
      execute: async () => {
        let image = await screenshot(delegate.container, screenshotOptions);
        return imageResponse(image);
      },
    },
  ];

  api.provideContext({ tools: tools });
}

function textResponse(text: string): ToolResponse {
  return { content: [{ type: "text", text: text }] };
}

function jsonResponse(content: any): ToolResponse {
  return textResponse(JSON.stringify(content));
}

function imageResponse(dataUrl: string): ToolResponse {
  let parsed = parseImageDataUrl(dataUrl);
  if (parsed) {
    return { content: [{ type: "image", data: parsed.data, mimeType: parsed.mimeType }] };
  }
  return textResponse("failed to take screenshot");
}

function parseImageDataUrl(dataUrl: string): { mimeType: string; data: string } | null {
  // Check if it's a valid data URL
  if (!dataUrl.startsWith("data:")) {
    return null;
  }

  // Find the comma that separates metadata from content
  const commaIndex = dataUrl.indexOf(",");
  if (commaIndex === -1) {
    return null;
  }

  // Extract the metadata part (everything before the comma)
  const metadata = dataUrl.substring(5, commaIndex); // Skip "data:"

  // Extract the base64 content (everything after the comma)
  const base64Content = dataUrl.substring(commaIndex + 1);

  // Parse the metadata to extract MIME type
  let mimeType: string;

  if (metadata.includes(";base64")) {
    // Format: "image/png;base64" or "image/jpeg;base64"
    mimeType = metadata.replace(";base64", "");
  } else if (metadata.includes(";")) {
    // Handle other parameters (though base64 is most common)
    mimeType = metadata.split(";")[0];
  } else {
    // Just the MIME type without parameters
    mimeType = metadata;
  }

  // Validate that it's an image MIME type
  if (!mimeType.startsWith("image/")) {
    return null;
  }

  // Specifically check for PNG and JPEG
  if (mimeType !== "image/png" && mimeType !== "image/jpeg") {
    return null;
  }

  return { mimeType, data: base64Content };
}
