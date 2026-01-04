// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import express from 'express';
import multer from 'multer';
import path from 'path';
import { fileURLToPath } from 'url';
import fs from 'fs/promises';
import { createProxyMiddleware } from 'http-proxy-middleware';
import { BackendManager } from './backend-manager.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = 3000;

// Storage for uploaded files
const upload = multer({
  dest: '/tmp/embedding-atlas-uploads/',
  limits: {
    fileSize: 1024 * 1024 * 1024, // 1GB limit
  }
});

// Backend manager
const backendManager = new BackendManager();
const logs = [];

// Store logs
backendManager.on('log', (message) => {
  logs.push({ type: 'log', message, timestamp: new Date().toISOString() });
  // Keep only last 100 logs
  if (logs.length > 100) {
    logs.shift();
  }
});

backendManager.on('error', (message) => {
  logs.push({ type: 'error', message, timestamp: new Date().toISOString() });
  if (logs.length > 100) {
    logs.shift();
  }
});

// Middleware
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

// API Routes

/**
 * Get backend status
 */
app.get('/api/status', (req, res) => {
  res.json({
    running: backendManager.isRunning(),
    url: backendManager.isRunning() ? backendManager.getUrl() : null,
    currentFile: backendManager.getCurrentFile(),
    logs: logs.slice(-20) // Last 20 logs
  });
});

/**
 * Upload and start backend with file
 */
app.post('/api/upload', upload.single('file'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No file uploaded' });
    }

    const filePath = req.file.path;
    console.log('[Launcher] Received file:', filePath);
    console.log('[Launcher] Original name:', req.file.originalname);

    // Parse options from request body
    const options = {
      port: 5055,
      host: 'localhost',
    };

    if (req.body.text) options.text = req.body.text;
    if (req.body.image) options.image = req.body.image;
    if (req.body.vector) options.vector = req.body.vector;
    if (req.body.x) options.x = req.body.x;
    if (req.body.y) options.y = req.body.y;

    // Clear logs
    logs.length = 0;

    // Start backend
    try {
      await backendManager.start(filePath, options);
      res.json({
        success: true,
        url: backendManager.getUrl(),
        message: 'Backend started successfully'
      });
    } catch (error) {
      console.error('[Launcher] Failed to start backend:', error);
      res.status(500).json({
        error: 'Failed to start backend',
        message: error.message,
        logs: logs
      });
    }
  } catch (error) {
    console.error('[Launcher] Error handling upload:', error);
    res.status(500).json({
      error: 'Upload failed',
      message: error.message
    });
  }
});

/**
 * Stop the backend
 */
app.post('/api/stop', async (req, res) => {
  try {
    await backendManager.stop();
    res.json({ success: true });
  } catch (error) {
    res.status(500).json({
      error: 'Failed to stop backend',
      message: error.message
    });
  }
});

/**
 * Get logs
 */
app.get('/api/logs', (req, res) => {
  res.json({ logs });
});

// Proxy to backend when running
app.use('/backend', (req, res, next) => {
  if (!backendManager.isRunning()) {
    return res.status(503).json({
      error: 'Backend not running',
      message: 'Please upload a file to start the backend'
    });
  }
  next();
}, createProxyMiddleware({
  target: 'http://localhost:5055',
  changeOrigin: true,
  pathRewrite: {
    '^/backend': ''
  },
  onError: (err, req, res) => {
    res.status(502).json({
      error: 'Backend proxy error',
      message: err.message
    });
  }
}));

// Serve launcher UI
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// Handle shutdown
process.on('SIGINT', async () => {
  console.log('\n[Launcher] Shutting down...');
  if (backendManager.isRunning()) {
    await backendManager.stop();
  }
  process.exit(0);
});

process.on('SIGTERM', async () => {
  console.log('\n[Launcher] Shutting down...');
  if (backendManager.isRunning()) {
    await backendManager.stop();
  }
  process.exit(0);
});

// Start server
app.listen(PORT, () => {
  console.log(`
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║             🚀 Embedding Atlas Launcher                   ║
║                                                           ║
║  Launcher UI: http://localhost:${PORT}                       ║
║                                                           ║
║  Upload a data file to start visualizing!                ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
  `);
});
