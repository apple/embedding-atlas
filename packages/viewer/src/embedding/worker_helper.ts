// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

export class WorkerRPC {
  static connect(worker: Worker): Promise<(name: string, ...args: any[]) => Promise<any>> {
    return new Promise((resolve) => {
      let callbacks = new Map<string, [any, any]>();
      let rpc = (name: string, ...args: any[]) => {
        return new Promise((resolve, reject) => {
          let id = new Date().getTime() + "-" + Math.random();
          callbacks.set(id, [resolve, reject]);
          worker.postMessage({ rpc: name, id: id, args: args });
        });
      };
      worker.postMessage({ ready: true });
      worker.onmessage = (e) => {
        if (e.data.ready) {
          resolve(rpc);
          return;
        }
        let cb = callbacks.get(e.data.id);
        if (cb != null) {
          callbacks.delete(e.data.id);
          if (e.data.error) {
            cb[1](new Error(e.data.error));
          } else {
            cb[0](e.data.result);
          }
        }
      };
    });
  }

  /** Call from worker */
  static runtime() {
    let methods = new Map<string, any>();
    let onmessage = async (ev: MessageEvent<any>) => {
      if (ev.data.ready) {
        postMessage({ ready: true });
      }
      if (ev.data.rpc) {
        let id = ev.data.id;
        let msg = {
          id: id,
          result: null,
          error: null,
        };
        try {
          msg.result = await methods.get(ev.data.rpc)?.(...ev.data.args);
        } catch (e: any) {
          msg.error = e.toString();
        }
        postMessage(msg);
      }
    };
    postMessage({ ready: true });
    return {
      handler: onmessage,
      register: (name: string, func: any) => {
        methods.set(name, func);
      },
    };
  }
}
