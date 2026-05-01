// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

// --- Shared types ---

/**
 * Proxy type that maps a class's methods to async RPC calls.
 * Only public methods are exposed; properties are excluded.
 * A `destroy()` method is always added to release the worker-side instance.
 */
export type WorkerProxy<T> = {
  [K in keyof T as T[K] extends (...args: any[]) => any ? K : never]: T[K] extends (...args: infer A) => infer R
    ? (...args: A) => Promise<Awaited<R>>
    : never;
} & { destroy(): Promise<void> };

// --- Internal protocol ---

const CB = "__cb__";

interface CallbackRef {
  [CB]: string;
}

function isCallbackRef(value: unknown): value is CallbackRef {
  return value != null && typeof value === "object" && typeof (value as any)[CB] === "string";
}

const TRANSFER = Symbol("transfer");

/**
 * Mark a value for zero-copy transfer across the worker boundary.
 * The underlying `ArrayBuffer`s are moved (not copied) via the structured clone
 * transfer list, so the sending side loses access after the call.
 *
 * Works on direct arguments and return values — not inside nested objects/arrays.
 * To transfer buffers nested in an object, wrap the whole object:
 *
 * ```ts
 * return transfer({ coords, indices }, [coords.buffer, indices.buffer]);
 * ```
 *
 * @param value    The value to send.
 * @param buffers  `ArrayBuffer`s to transfer (e.g. `[typedArray.buffer]`).
 * @returns An opaque wrapper typed as `T` for ergonomic use in args / return values.
 */
export function transfer<T>(value: T, buffers: ArrayBuffer[]): T {
  return { [TRANSFER]: buffers, value } as any;
}

function unwrapTransfer(value: any): [any, ArrayBuffer[]] {
  if (value != null && typeof value === "object" && TRANSFER in value) {
    return [value.value, value[TRANSFER]];
  }
  return [value, []];
}

let _id = 0;
function nextId(): string {
  return String(++_id);
}

// --- Main thread API ---

/**
 * Main-thread handle to a connected worker.
 *
 * Obtained via {@link connectWorker}. Provides two ways to call into the worker:
 * - `create()` — instantiate a registered class remotely and get a {@link WorkerProxy} back.
 * - `call()` — invoke a registered plain function by name.
 *
 * Function-typed arguments are automatically converted to callbacks: the worker
 * can call them and the main-thread function executes (fire-and-forget, no return value).
 */
export interface WorkerConnection {
  /** Create a class instance on the worker and return a typed proxy. */
  create<T>(className: string, ...args: any[]): Promise<WorkerProxy<T>>;
  /** Call a registered function on the worker. */
  call(name: string, ...args: any[]): Promise<any>;
}

/**
 * Connect to a web worker and return a {@link WorkerConnection}.
 *
 * Performs a ready-handshake with the worker (which must call
 * {@link createWorkerRuntime}) and resolves once the worker signals it is ready.
 *
 * @param worker  A `Worker` instance (e.g. `new Worker(new URL("./my.worker.js", import.meta.url), { type: "module" })`).
 * @returns A promise that resolves to a {@link WorkerConnection} once the handshake completes.
 */
export function connectWorker(worker: Worker): Promise<WorkerConnection> {
  return new Promise((resolve) => {
    const pending = new Map<string, { resolve(v: any): void; reject(e: Error): void }>();
    const callbacks = new Map<string, (...args: any[]) => any>();

    function send(msg: Record<string, any>, callArgs?: any[]): Promise<any> {
      return new Promise((res, rej) => {
        const id = nextId();
        const cbIds: string[] = [];
        const transfers: ArrayBuffer[] = [];

        function processArg(arg: any): any {
          if (typeof arg === "function") {
            const cbId = nextId();
            cbIds.push(cbId);
            callbacks.set(cbId, arg);
            return { [CB]: cbId };
          }
          const [val, bufs] = unwrapTransfer(arg);
          if (bufs.length > 0) {
            transfers.push(...bufs);
            return val;
          }
          if (val != null && typeof val === "object" && Object.getPrototypeOf(val) === Object.prototype) {
            const result: Record<string, any> = {};
            for (const key of Object.keys(val)) {
              result[key] = processArg(val[key]);
            }
            return result;
          }
          return val;
        }

        let processedArgs: any[] | undefined;
        if (callArgs) {
          processedArgs = callArgs.map(processArg);
        }

        function cleanup() {
          for (const cbId of cbIds) callbacks.delete(cbId);
        }

        pending.set(id, {
          resolve(v) {
            cleanup();
            res(v);
          },
          reject(e) {
            cleanup();
            rej(e);
          },
        });

        worker.postMessage({ ...msg, id, args: processedArgs }, { transfer: transfers });
      });
    }

    worker.postMessage({ type: "ready" });
    worker.onmessage = (e: MessageEvent) => {
      const { data } = e;
      switch (data.type) {
        case "ready":
          resolve(connection);
          break;

        case "callback": {
          const cb = callbacks.get(data.callbackId);
          if (cb) cb(...data.args);
          break;
        }

        case "result": {
          const p = pending.get(data.id);
          if (!p) break;
          pending.delete(data.id);
          if (data.error != null) p.reject(new Error(data.error));
          else p.resolve(data.result);
          break;
        }
      }
    };

    function makeProxy<T>(handle: string): WorkerProxy<T> {
      return new Proxy({} as WorkerProxy<T>, {
        get(_target, method: string | symbol) {
          if (method === "then" || typeof method === "symbol") return undefined;
          if (method === "destroy") return () => send({ type: "destroy", handle });
          return (...args: any[]) => send({ type: "call", handle, method: String(method) }, args);
        },
      });
    }

    const connection: WorkerConnection = {
      async create<T>(className: string, ...args: any[]): Promise<WorkerProxy<T>> {
        const handle = await send({ type: "create", className }, args);
        return makeProxy<T>(handle);
      },
      call(name: string, ...args: any[]): Promise<any> {
        return send({ type: "callFn", name }, args);
      },
    };
  });
}

// --- Worker thread API ---

/**
 * Worker-side runtime returned by {@link createWorkerRuntime}.
 *
 * Provides a message `handler` (assign to `onmessage`) and registration
 * functions for classes and plain functions that become callable from the
 * main thread via {@link WorkerConnection}.
 */
export interface WorkerRuntime {
  /** Assign to `onmessage` in the worker. */
  handler: (ev: MessageEvent) => void;
  /** Register a class factory. The factory receives constructor args and returns an instance (or Promise). */
  registerClass(name: string, factory: (...args: any[]) => any): void;
  /** Register a plain function callable from the main thread. */
  registerFunction(name: string, fn: (...args: any[]) => any): void;
}

/**
 * Initialize the worker-side RPC runtime.
 *
 * Call this once at the top of a web worker module, assign the returned
 * `handler` to `onmessage`, and register classes/functions:
 *
 * ```ts
 * const { handler, registerClass, registerFunction } = createWorkerRuntime();
 * onmessage = handler;
 *
 * registerClass("Embedder", (opts) => Embedder.create(opts));
 * registerFunction("ping", () => "pong");
 * ```
 *
 * @returns A {@link WorkerRuntime} with the message handler and registration helpers.
 */
export function createWorkerRuntime(): WorkerRuntime {
  const functions = new Map<string, (...args: any[]) => any>();
  const factories = new Map<string, (...args: any[]) => any>();
  const instances = new Map<string, any>();
  function resolveArg(arg: any): any {
    if (isCallbackRef(arg)) {
      const callbackId = arg[CB];
      return (...cbArgs: any[]) => {
        const transfers: ArrayBuffer[] = [];
        const processed = cbArgs.map((a) => {
          const [val, bufs] = unwrapTransfer(a);
          transfers.push(...bufs);
          return val;
        });
        postMessage({ type: "callback", callbackId, args: processed }, { transfer: transfers });
      };
    }
    if (arg != null && typeof arg === "object" && Object.getPrototypeOf(arg) === Object.prototype) {
      const result: Record<string, any> = {};
      for (const key of Object.keys(arg)) {
        result[key] = resolveArg(arg[key]);
      }
      return result;
    }
    return arg;
  }

  function resolveArgs(args: any[]): any[] {
    return args.map(resolveArg);
  }

  const handler = async (ev: MessageEvent) => {
    const { data } = ev;

    if (data.type === "ready") {
      postMessage({ type: "ready" });
      return;
    }

    const id: string = data.id;
    let result: any = undefined;
    let error: string | null = null;

    try {
      const args = data.args ? resolveArgs(data.args) : [];

      switch (data.type) {
        case "create": {
          const factory = factories.get(data.className);
          if (!factory) throw new Error(`Unknown class: ${data.className}`);
          const instance = await factory(...args);
          const handle = nextId();
          instances.set(handle, instance);
          result = handle;
          break;
        }
        case "call": {
          const instance = instances.get(data.handle);
          if (!instance) throw new Error(`Unknown handle: ${data.handle}`);
          const method = instance[data.method];
          if (typeof method !== "function") throw new Error(`Not a method: ${data.method}`);
          result = await method.call(instance, ...args);
          break;
        }
        case "callFn": {
          const fn = functions.get(data.name);
          if (!fn) throw new Error(`Unknown function: ${data.name}`);
          result = await fn(...args);
          break;
        }
        case "destroy": {
          const instance = instances.get(data.handle);
          if (instance) {
            instances.delete(data.handle);
            instance.destroy?.();
          }
          break;
        }
      }
    } catch (e: any) {
      error = String(e);
    }

    const [unwrappedResult, transfers] = unwrapTransfer(result);
    postMessage({ type: "result", id, result: unwrappedResult, error }, { transfer: transfers });
  };

  postMessage({ type: "ready" });

  return {
    handler,
    registerClass(name: string, factory: (...args: any[]) => any) {
      factories.set(name, factory);
    },
    registerFunction(name: string, fn: (...args: any[]) => any) {
      functions.set(name, fn);
    },
  };
}
