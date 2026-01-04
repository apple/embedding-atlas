# Embedding Atlas Launcher

Web-based launcher application that provides a GUI for uploading and visualizing data files with Embedding Atlas, without requiring terminal access.

## Features

- **📁 File Upload**: Drag & drop or browse to select data files (Parquet, CSV, JSON, JSONL)
- **🚀 Automatic Backend Management**: Automatically starts and manages the Python backend process
- **📊 Integrated Viewer**: Seamlessly proxies to the visualization interface
- **🔄 Process Control**: Start, stop, and restart the backend with a single click
- **📝 Real-time Logs**: View backend logs and status in real-time
- **🌐 Web-based**: No Electron or native dependencies required

## Architecture

The launcher is a lightweight Node.js Express server that:

1. Serves a web-based UI for file upload
2. Manages Python backend process lifecycle
3. Proxies requests to the backend once it's running
4. Provides real-time status and logs via REST API

```
┌──────────────────┐
│   Browser        │
│  (Launcher UI)   │
└────────┬─────────┘
         │
         │ HTTP
         │
┌────────▼─────────┐
│  Node.js Server  │  ◄─── Manages Python process
│   (Express)      │  ◄─── File uploads
└────────┬─────────┘  ◄─── Proxies to backend
         │
         │ Child Process
         │
┌────────▼─────────┐
│ Python Backend   │
│ (embedding-atlas)│
│  FastAPI Server  │
└──────────────────┘
```

## Prerequisites

1. **Python Backend Installed**: The `embedding-atlas` Python package must be installed and available in your PATH
   ```bash
   # Install from the backend package
   cd packages/backend
   uv build --wheel
   pip install dist/*.whl

   # Verify installation
   embedding-atlas --version
   ```

2. **Node.js Dependencies**: Install workspace dependencies
   ```bash
   # From the repository root
   npm install
   ```

## Usage

### Quick Start

From the repository root:

```bash
npm run launcher
```

Then open your browser to: **http://localhost:3000**

### Development Mode

Run with auto-reload on file changes:

```bash
npm run launcher:dev
```

### From the Launcher Package

```bash
cd packages/launcher
npm start
```

## How It Works

### User Flow

1. **Open Launcher**: Navigate to http://localhost:3000
2. **Upload File**: Drag & drop or browse to select your data file
3. **Start Backend**: Click "Start Visualization"
4. **View Data**: Once the backend is ready, click "Open Viewer" to visualize your data
5. **Stop/Restart**: Use the controls to stop the backend or upload a new file

### File Handling

When you upload a file:
1. File is saved to `/tmp/embedding-atlas-uploads/`
2. Node.js spawns: `embedding-atlas <file> --port 5055 --host localhost`
3. Backend process starts and begins loading data
4. Health checks poll `http://localhost:5055/data/metadata.json`
5. Once healthy, UI shows "Open Viewer" link
6. Clicking the link opens `/backend/` which proxies to `http://localhost:5055/`

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serves launcher UI |
| `/api/status` | GET | Get backend status, logs, current file |
| `/api/upload` | POST | Upload file and start backend |
| `/api/stop` | POST | Stop the backend process |
| `/api/logs` | GET | Get all backend logs |
| `/backend/*` | * | Proxy to running backend (port 5055) |

## Project Structure

```
packages/launcher/
├── src/
│   ├── server.js              # Express server
│   ├── backend-manager.js     # Python process lifecycle manager
│   └── public/
│       ├── index.html         # Launcher UI
│       └── app.js             # Frontend JavaScript
├── package.json
└── README.md
```

## Configuration

### Default Settings

- **Launcher Port**: 3000 (configurable in `server.js`)
- **Backend Port**: 5055 (hardcoded to match backend default)
- **Upload Directory**: `/tmp/embedding-atlas-uploads/`
- **File Size Limit**: 1GB

### Customizing Backend Options

To pass additional options to the backend (e.g., `--text`, `--image`), modify the upload handler in `server.js`:

```javascript
const options = {
  port: 5055,
  host: 'localhost',
  text: 'column_name',  // Add your custom options here
};
```

Or extend the UI to include form inputs for these options.

## Troubleshooting

### Backend command not found

**Error**: `ENOENT: embedding-atlas command not found`

**Solution**:
1. Ensure `embedding-atlas` is installed: `pip install embedding-atlas`
2. Verify it's in PATH: `which embedding-atlas` (Unix) or `where embedding-atlas` (Windows)
3. If using a virtual environment, ensure it's activated

### Port already in use

**Error**: `EADDRINUSE: address already in use :::3000`

**Solution**:
- Change the launcher port in `server.js`:
  ```javascript
  const PORT = 3001; // Or any available port
  ```

### Backend fails to start

Check the logs displayed in the launcher UI. Common issues:

- **Missing dependencies**: Install required Python packages
- **Invalid file format**: Ensure file is valid Parquet/CSV/JSON
- **Memory issues**: Large files may require more RAM
- **Permission denied**: Ensure write access to `/tmp/embedding-atlas-uploads/`

### File upload fails

**Error**: `File too large` or `413 Payload Too Large`

**Solution**:
- Increase the file size limit in `server.js`:
  ```javascript
  const upload = multer({
    dest: '/tmp/embedding-atlas-uploads/',
    limits: {
      fileSize: 2 * 1024 * 1024 * 1024, // 2GB
    }
  });
  ```

## Advantages Over Electron

This approach provides several benefits:

- ✅ **Lightweight**: No large Electron binaries to download
- ✅ **Cross-platform**: Works anywhere Node.js runs
- ✅ **Browser-based**: Use any modern browser
- ✅ **Simple deployment**: Just `npm run launcher`
- ✅ **Easy debugging**: Standard browser DevTools
- ✅ **No build step**: Plain HTML/CSS/JS for the UI

## Development

### Adding Features

**Backend Options**:
- Edit `backend-manager.js` to add new CLI arguments
- Extend the `start()` method options object

**UI Enhancements**:
- Modify `public/index.html` for layout changes
- Edit `public/app.js` for functionality changes
- Use standard CSS for styling

**API Extensions**:
- Add new endpoints in `server.js`
- Follow the existing REST API pattern

### Testing

1. Start the launcher: `npm run launcher`
2. Open http://localhost:3000
3. Upload a test file (e.g., a small Parquet file)
4. Verify backend starts and viewer loads
5. Test stop/restart functionality
6. Check logs display correctly

## Future Enhancements

Potential improvements:

- [ ] Support for backend configuration options in UI (text column, model selection, etc.)
- [ ] Multiple file uploads and dataset management
- [ ] Persistent backend sessions across browser refreshes
- [ ] WebSocket for real-time log streaming
- [ ] Built-in dataset samples for quick testing
- [ ] Backend resource usage monitoring

## License

Copyright (c) 2025 Apple Inc. Licensed under MIT License.
