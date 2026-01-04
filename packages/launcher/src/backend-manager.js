// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { spawn } from 'child_process';
import { EventEmitter } from 'events';
import http from 'http';

/**
 * Manages the Python backend process lifecycle
 */
export class BackendManager extends EventEmitter {
  constructor() {
    super();
    this.process = null;
    this.port = 5055;
    this.host = 'localhost';
    this.ready = false;
    this.currentFile = null;
  }

  /**
   * Start the backend with a data file
   * @param {string} filePath - Path to the data file
   * @param {object} options - Backend options
   */
  async start(filePath, options = {}) {
    // Stop existing process if running
    if (this.process) {
      await this.stop();
    }

    const args = [filePath];

    // Add optional arguments
    if (options.text) args.push('--text', options.text);
    if (options.image) args.push('--image', options.image);
    if (options.vector) args.push('--vector', options.vector);
    if (options.x) args.push('--x', options.x);
    if (options.y) args.push('--y', options.y);
    if (options.port) {
      this.port = options.port;
    }
    args.push('--port', this.port.toString());

    if (options.host) {
      this.host = options.host;
    }
    args.push('--host', this.host);

    // Disable auto port finding
    args.push('--no-auto-port');

    console.log('[Backend] Starting with command: embedding-atlas', args.join(' '));

    // Spawn the Python backend process
    this.process = spawn('embedding-atlas', args, {
      stdio: ['ignore', 'pipe', 'pipe'],
      env: { ...process.env }
    });

    this.currentFile = filePath;

    // Handle stdout
    this.process.stdout.on('data', (data) => {
      const message = data.toString();
      console.log('[Backend]', message);
      this.emit('log', message);

      // Check for startup message
      if (message.includes('Uvicorn running on')) {
        this.ready = true;
        this.emit('ready');
      }
    });

    // Handle stderr
    this.process.stderr.on('data', (data) => {
      const message = data.toString();
      console.error('[Backend Error]', message);
      this.emit('error', message);
    });

    // Handle process exit
    this.process.on('exit', (code, signal) => {
      console.log(`[Backend] Process exited with code ${code} and signal ${signal}`);
      this.ready = false;
      this.process = null;
      this.currentFile = null;
      this.emit('exit', { code, signal });
    });

    // Handle process errors
    this.process.on('error', (err) => {
      console.error('[Backend] Failed to start:', err);
      this.emit('spawn-error', err.message);
      this.process = null;
      this.currentFile = null;
    });

    // Wait for backend to be ready
    return this.waitForReady();
  }

  /**
   * Wait for the backend to be ready by polling the health endpoint
   */
  async waitForReady(timeout = 60000) {
    const startTime = Date.now();
    const pollInterval = 500;

    while (Date.now() - startTime < timeout) {
      if (await this.checkHealth()) {
        this.ready = true;
        this.emit('ready');
        return true;
      }
      await new Promise(resolve => setTimeout(resolve, pollInterval));
    }

    throw new Error('Backend failed to start within timeout');
  }

  /**
   * Check if backend is healthy
   */
  async checkHealth() {
    return new Promise((resolve) => {
      const req = http.get(`http://${this.host}:${this.port}/data/metadata.json`, (res) => {
        resolve(res.statusCode === 200);
      });

      req.on('error', () => {
        resolve(false);
      });

      req.setTimeout(1000, () => {
        req.destroy();
        resolve(false);
      });
    });
  }

  /**
   * Stop the backend process
   */
  async stop() {
    if (!this.process) {
      return;
    }

    return new Promise((resolve) => {
      const timeout = setTimeout(() => {
        if (this.process) {
          console.log('[Backend] Force killing process');
          this.process.kill('SIGKILL');
        }
      }, 5000);

      this.process.once('exit', () => {
        clearTimeout(timeout);
        this.ready = false;
        this.process = null;
        this.currentFile = null;
        console.log('[Backend] Stopped');
        resolve();
      });

      // Try graceful shutdown first
      console.log('[Backend] Stopping...');
      this.process.kill('SIGTERM');
    });
  }

  /**
   * Get the backend URL
   */
  getUrl() {
    return `http://${this.host}:${this.port}`;
  }

  /**
   * Check if backend is running
   */
  isRunning() {
    return this.process !== null && this.ready;
  }

  /**
   * Get current file
   */
  getCurrentFile() {
    return this.currentFile;
  }
}
