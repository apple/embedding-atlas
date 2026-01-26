import { nanoid } from "nanoid";
import { writable, type Writable } from "svelte/store";

/** A log message for the data importing UI */
export interface LogMessage {
  text: string;
  markdown?: boolean;
  progress?: number;
  progressText?: string;
  error?: boolean;
}

export interface MessageHandle {
  update: (value: Partial<LogMessage>) => void;
}

export class Logger {
  messages: Writable<(LogMessage & { _id: string })[]>;

  constructor() {
    this.messages = writable([]);
  }

  private append(message: LogMessage): MessageHandle {
    let id = nanoid();

    this.messages.update((target) => [...target, { ...message, _id: id }]);

    return {
      update: (value) => {
        this.messages.update((target) => {
          return target.map((msg) => {
            if (msg._id == id) {
              return { ...msg, ...value };
            } else {
              return msg;
            }
          });
        });
      },
    };
  }

  info(text: string, options: Omit<LogMessage, "text"> = {}): MessageHandle {
    return this.append({ text: text, ...options });
  }

  error(text: string, options: Omit<LogMessage, "text"> = {}): MessageHandle {
    return this.append({ text: text, ...options, error: true });
  }

  exception(exception: unknown): MessageHandle {
    if (exception instanceof LoggableError) {
      return this.append({ text: exception.message, ...exception.loggerOptions, error: true });
    } else if (exception instanceof Error) {
      return this.append({ text: exception.message, error: true });
    } else {
      return this.append({ text: String(exception), error: true });
    }
  }
}

export class LoggableError extends Error {
  loggerOptions: Omit<LogMessage, "text">;

  constructor(message?: string, options: Omit<LogMessage, "text"> = {}) {
    super(message);
    this.loggerOptions = options;
  }
}
