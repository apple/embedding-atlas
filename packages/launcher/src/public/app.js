// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

// Elements
const uploadZone = document.getElementById('uploadZone');
const fileInput = document.getElementById('fileInput');
const selectedFileDiv = document.getElementById('selectedFile');
const fileName = document.getElementById('fileName');
const fileSize = document.getElementById('fileSize');
const clearFileButton = document.getElementById('clearFile');
const startButton = document.getElementById('startButton');
const stopButton = document.getElementById('stopButton');
const uploadSection = document.getElementById('uploadSection');
const loadingSection = document.getElementById('loadingSection');
const runningSection = document.getElementById('runningSection');
const logsSection = document.getElementById('logsSection');
const logsDiv = document.getElementById('logs');
const statusDiv = document.getElementById('status');
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const currentFileSpan = document.getElementById('currentFile');
const stopAndRestartButton = document.getElementById('stopAndRestart');

let selectedFile = null;
let statusPollInterval = null;

// Initialize
checkStatus();
startStatusPolling();

// File selection handlers
uploadZone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', handleFileSelect);
clearFileButton.addEventListener('click', clearFile);
startButton.addEventListener('click', startBackend);
stopButton.addEventListener('click', stopBackend);
stopAndRestartButton.addEventListener('click', stopAndRestart);

// Drag and drop
uploadZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  uploadZone.classList.add('drag-over');
});

uploadZone.addEventListener('dragleave', () => {
  uploadZone.classList.remove('drag-over');
});

uploadZone.addEventListener('drop', (e) => {
  e.preventDefault();
  uploadZone.classList.remove('drag-over');
  const files = e.dataTransfer.files;
  if (files.length > 0) {
    handleFile(files[0]);
  }
});

function handleFileSelect(e) {
  const files = e.target.files;
  if (files.length > 0) {
    handleFile(files[0]);
  }
}

function handleFile(file) {
  selectedFile = file;
  fileName.textContent = file.name;
  fileSize.textContent = formatFileSize(file.size);
  selectedFileDiv.classList.remove('hidden');
  startButton.disabled = false;
}

function clearFile() {
  selectedFile = null;
  fileInput.value = '';
  selectedFileDiv.classList.add('hidden');
  startButton.disabled = true;
}

function formatFileSize(bytes) {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

async function startBackend() {
  if (!selectedFile) return;

  // Show loading
  uploadSection.classList.add('hidden');
  runningSection.classList.add('hidden');
  loadingSection.classList.remove('hidden');
  logsSection.classList.remove('hidden');

  const formData = new FormData();
  formData.append('file', selectedFile);

  try {
    const response = await fetch('/api/upload', {
      method: 'POST',
      body: formData
    });

    const result = await response.json();

    if (response.ok && result.success) {
      // Success - start polling for status
      setTimeout(checkStatus, 1000);
    } else {
      // Error
      alert(`Failed to start backend: ${result.message || result.error}`);
      loadingSection.classList.add('hidden');
      uploadSection.classList.remove('hidden');
      if (result.logs) {
        displayLogs(result.logs);
      }
    }
  } catch (error) {
    alert(`Error: ${error.message}`);
    loadingSection.classList.add('hidden');
    uploadSection.classList.remove('hidden');
  }
}

async function stopBackend() {
  try {
    await fetch('/api/stop', { method: 'POST' });
    checkStatus();
  } catch (error) {
    alert(`Error stopping backend: ${error.message}`);
  }
}

async function stopAndRestart() {
  await stopBackend();
  setTimeout(() => {
    runningSection.classList.add('hidden');
    uploadSection.classList.remove('hidden');
    logsSection.classList.add('hidden');
    logsDiv.innerHTML = '';
  }, 500);
}

async function checkStatus() {
  try {
    const response = await fetch('/api/status');
    const status = await response.json();

    if (status.running) {
      // Backend is running
      statusDot.classList.add('running');
      statusText.textContent = 'Backend running';
      statusDiv.classList.add('running');
      stopButton.classList.remove('hidden');

      // Show running section
      loadingSection.classList.add('hidden');
      uploadSection.classList.add('hidden');
      runningSection.classList.remove('hidden');
      currentFileSpan.textContent = status.currentFile || 'Unknown';

      // Display logs
      if (status.logs && status.logs.length > 0) {
        displayLogs(status.logs);
      }
    } else {
      // Backend not running
      statusDot.classList.remove('running');
      statusText.textContent = 'Backend not running';
      statusDiv.classList.remove('running');
      stopButton.classList.add('hidden');

      // Show upload section if not loading
      if (!loadingSection.classList.contains('hidden')) {
        // Still loading, keep checking
      } else {
        runningSection.classList.add('hidden');
        if (uploadSection.classList.contains('hidden')) {
          uploadSection.classList.remove('hidden');
        }
      }
    }
  } catch (error) {
    console.error('Error checking status:', error);
  }
}

function displayLogs(logs) {
  if (!Array.isArray(logs)) return;

  const logsHtml = logs.map(log => {
    const className = log.type === 'error' ? 'log-entry error' : 'log-entry';
    return `<div class="${className}">[${new Date(log.timestamp).toLocaleTimeString()}] ${escapeHtml(log.message)}</div>`;
  }).join('');

  logsDiv.innerHTML = logsHtml;
  logsDiv.scrollTop = logsDiv.scrollHeight;
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

function startStatusPolling() {
  statusPollInterval = setInterval(checkStatus, 2000);
}

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
  if (statusPollInterval) {
    clearInterval(statusPollInterval);
  }
});
