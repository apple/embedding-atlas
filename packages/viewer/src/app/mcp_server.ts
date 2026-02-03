import { type JSONSchema7 } from "json-schema";
import { writable, type Readable, type Writable } from "svelte/store";
import { ReconnectingWebSocket } from "./reconnecting_websocket.js";

/** A type that mirrors the current design in the upcoming navigator.modelContext API */
export interface ModelContextAPI {
  provideContext(context: MCPContext): void;

  readonly connectionStatus?: Readable<"connecting" | "connected" | "closed" | "error">;
}

export interface MCPContext {
  tools?: MCPTool[];
}

/** Tool definition interface */
export interface MCPTool {
  /** Unique name for the tool */
  name: string;

  /** The title of the tool */
  title?: string;

  /** Natural language description of what the tool does */
  description: string;

  /** JSON Schema defining the input parameters */
  inputSchema: JSONSchema7;

  /** JSON Schema defining the output parameters */
  outputSchema?: JSONSchema7;

  /** Function that implements the tool and returns a result */
  execute: (input: any, agent: unknown) => Promise<ToolResponse>;
}

/** Tool response format */
export interface ToolResponse {
  content: Array<{
    type: "text" | "image" | "video";
    text?: string;
    url?: string;
    [key: string]: any;
  }>;
  isError?: boolean;
}

interface WSRequest {
  id: string;
  request: JSONRPCRequest | JSONRPCRequest[];
}

interface WSResponse {
  id: string;
  response: JSONRPCResponse | JSONRPCResponse[];
}

interface JSONRPCRequest {
  jsonrpc: "2.0";
  id: number;
  method: string;
  params?: Record<string, any>;
}

interface JSONRPCResponse {
  jsonrpc: "2.0";
  id: number;
  result?: any;
  error?: any;
}

export class MCPWebSocketServer implements ModelContextAPI {
  private tools: MCPTool[];
  private toolsMap: Map<string, MCPTool>;
  private ws: ReconnectingWebSocket;

  connectionStatus: Writable<"connecting" | "connected" | "closed" | "error">;

  constructor(endpoint: string) {
    this.tools = [];
    this.toolsMap = new Map();
    this.connectionStatus = writable("connecting");

    this.ws = new ReconnectingWebSocket(endpoint, {
      onMessage: async (event) => {
        try {
          const data = JSON.parse(event.data);

          // Check for control messages
          if (data.control === "close") {
            this.ws.close();
            return;
          }

          // Handle normal MCP requests
          const request: WSRequest = data;
          const response = await this.processRequest(request);
          this.ws.send(JSON.stringify(response));
        } catch (error) {
          console.error("Error processing MCP request:", error);
        }
      },
      onStatus: (value) => {
        this.connectionStatus.set(value);
      },
    });
  }

  public close(): void {
    this.ws.close();
  }

  provideContext(context: MCPContext) {
    this.tools = context.tools ?? [];
    this.toolsMap = new Map(this.tools.map((tool) => [tool.name, tool]));
  }

  private async processRequest(request: WSRequest): Promise<WSResponse> {
    if (request.request instanceof Array) {
      let responses: JSONRPCResponse[] = [];
      for (let req of request.request) {
        responses.push(await this.processRPC(req));
      }
      return { id: request.id, response: responses };
    } else {
      return { id: request.id, response: await this.processRPC(request.request) };
    }
  }

  private async processRPC(request: JSONRPCRequest): Promise<JSONRPCResponse> {
    let result: any;
    let error: any;
    try {
      result = await this.rpc(request.method, request.params ?? {});
    } catch (e: any) {
      error = { code: -32603, message: e.toString() };
    }
    return {
      jsonrpc: "2.0",
      id: request.id,
      result: result,
      error: error,
    };
  }

  private async rpc(method: string, params: any): Promise<any> {
    switch (method) {
      case "initialize":
        return await this.initialize(params);
      case "tools/list":
        return await this.toolsList(params);
      case "tools/call":
        return await this.toolsCall(params);
      default:
        throw new Error(`method ${method} is not implemented`);
    }
  }

  async initialize(params: { clientInfo: { name: string } }) {
    console.info("MCP Initialize", params.clientInfo);
    return {
      protocolVersion: "2024-11-05",
      capabilities: {
        logging: {},
        prompts: {},
        resources: {},
        tools: {},
        tasks: {},
      },
      serverInfo: {
        name: "Embedding Atlas",
        title: "Embedding Atlas MCP Server",
        version: "1.0.0",
        description: "MCP server for the Embedding Atlas frontend",
        icons: [
          {
            src: "https://apple.github.io/embedding-atlas/favicon.svg",
            mimeType: "image/svg+xml",
            sizes: ["any"],
          },
        ],
        websiteUrl: "https://apple.github.io/embedding-atlas",
      },
      instructions: "Optional instructions for the client",
    };
  }

  async toolsList(params: {}) {
    let resultTools = this.tools.map((tool) => ({
      name: tool.name,
      title: tool.title,
      description: tool.description,
      inputSchema: tool.inputSchema,
      outputSchema: tool.outputSchema,
    }));

    return { tools: resultTools };
  }

  async toolsCall(params: { name: string; arguments: any }): Promise<ToolResponse> {
    let tool = this.toolsMap.get(params.name);
    if (tool == undefined) {
      throw new Error("tool not found");
    }
    try {
      return await tool.execute(params.arguments, undefined);
    } catch (e: any) {
      return {
        content: [{ type: "text", text: "Exception: " + e.toString() }],
        isError: true,
      };
    }
  }
}
