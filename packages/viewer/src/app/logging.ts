import { writable, type Writable } from "svelte/store";

/** A log message for the data importing UI */
export interface LogMessage {
  text: string;
  markdown?: boolean;
  progress?: number;
  progressText?: string;
  error?: boolean;
}

export class Logger {
  messages: Writable<LogMessage[]>;

  constructor() {
    this.messages = writable([]);
  }

  private append(message: LogMessage) {
    this.messages.update((target) => {
      if (target.length > 0 && target[target.length - 1].text == message.text) {
        return [...target.slice(0, target.length - 1), message];
      } else {
        return [...target, message];
      }
    });
  }

  info(text: string, options: Omit<LogMessage, "text"> = {}) {
    this.append({ text: text, ...options });
  }

  error(text: string, options: Omit<LogMessage, "text"> = {}) {
    this.append({ text: text, ...options, error: true });
  }

  exception(exception: unknown) {
    if (exception instanceof LoggableError) {
      this.append({ text: exception.message, ...exception.loggerOptions, error: true });
    } else if (exception instanceof Error) {
      this.append({ text: exception.message, error: true });
    } else {
      this.append({ text: String(exception), error: true });
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
