export interface ReconnectingWebSocketOptions {
  maxReconnectAttempts?: number;
  initialReconnectDelay?: number;
  maxReconnectDelay?: number;
  onMessage: (event: MessageEvent) => void;
  onStatus?: (status: "connecting" | "connected" | "closed" | "error") => void;
}

export class ReconnectingWebSocket {
  private ws: WebSocket | null = null;
  private endpoint: string;
  private shouldReconnect: boolean = true;
  private reconnectAttempts: number = 0;
  private maxReconnectAttempts: number;
  private reconnectDelay: number;
  private maxReconnectDelay: number;
  private reconnectTimeoutId: number | null = null;
  private onMessage: (event: MessageEvent) => void;
  private onStatus?: (status: "connecting" | "connected" | "closed" | "error") => void;

  constructor(endpoint: string, options: ReconnectingWebSocketOptions) {
    this.endpoint = endpoint;
    this.maxReconnectAttempts = options.maxReconnectAttempts ?? 10;
    this.reconnectDelay = options.initialReconnectDelay ?? 1000;
    this.maxReconnectDelay = options.maxReconnectDelay ?? 30000;
    this.onMessage = options.onMessage;
    this.onStatus = options.onStatus;
    this.connect();
  }

  private connect(): void {
    if (!this.shouldReconnect) {
      return;
    }

    this.onStatus?.("connecting");

    try {
      this.ws = new WebSocket(this.endpoint);

      this.ws.onopen = () => {
        console.debug("WebSocket connected");
        this.reconnectAttempts = 0;
        this.onStatus?.("connected");
        // Reset delay to initial value on successful connection
      };

      this.ws.onmessage = (event) => {
        this.onMessage(event);
      };

      this.ws.onclose = (event) => {
        console.debug("WebSocket closed:", event.code, event.reason);
        this.ws = null;

        if (this.shouldReconnect && this.reconnectAttempts < this.maxReconnectAttempts) {
          this.scheduleReconnect();
        } else if (this.reconnectAttempts >= this.maxReconnectAttempts) {
          console.error("Max reconnection attempts reached, giving up");
          this.onStatus?.("error");
        }
      };

      this.ws.onerror = (error) => {
        console.error("WebSocket error:", error);
      };
    } catch (error) {
      console.error("Failed to create WebSocket:", error);
      if (this.shouldReconnect && this.reconnectAttempts < this.maxReconnectAttempts) {
        this.scheduleReconnect();
      } else {
        this.onStatus?.("error");
      }
    }
  }

  private scheduleReconnect(): void {
    if (this.reconnectTimeoutId) {
      clearTimeout(this.reconnectTimeoutId);
    }

    this.reconnectAttempts++;
    const delay = Math.min(this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1), this.maxReconnectDelay);

    console.debug(`Scheduling reconnection attempt ${this.reconnectAttempts} in ${delay}ms`);

    this.reconnectTimeoutId = window.setTimeout(() => {
      this.reconnectTimeoutId = null;
      this.connect();
    }, delay);
  }

  public send(data: string): boolean {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(data);
      return true;
    }
    return false;
  }

  public close(): void {
    this.shouldReconnect = false;
    if (this.reconnectTimeoutId) {
      clearTimeout(this.reconnectTimeoutId);
      this.reconnectTimeoutId = null;
    }
    this.ws?.close();
    this.onStatus?.("closed");
  }
}
