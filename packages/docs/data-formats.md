# Data Formats

## Input Data

Embedding Atlas supports loading data from the following file formats:

- **Parquet** (`.parquet`)
- **JSONL** (`.jsonl`) — one JSON object per line
- **CSV** (`.csv`)

When using the [Python Notebook Widget](./widget.md) or [Streamlit Component](./streamlit.md), you can pass a **pandas DataFrame** directly.

## Column Display Types

Embedding Atlas provides several built-in renderers for displaying column values in the tooltip, instances view, and search results:

| Renderer          | Description                                                                                                                |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `markdown`        | Render the value as Markdown.                                                                                              |
| `liquid-template` | Render the value with a [Liquid](https://liquidjs.com/) template. Options: `template` (string), defaults to `{{ value }}`. |
| `image`           | Render the value as an image. Options: `size` (number), the max width/height in pixels.                                    |
| `audio`           | Render the value as an audio player.                                                                                       |
| `url`             | Render the value as a clickable link.                                                                                      |
| `json`            | Render the value as formatted JSON.                                                                                        |
| `messages`        | Render the value as chat messages (OpenAI format).                                                                         |

## Image Data

Embedding Atlas can display images in tooltips and the instances view. Image column values can be provided in the following formats:

- **URL**: A string starting with `http://` or `https://` pointing to the image.
- **Data URL**: A string starting with `data:image/...` containing inline image data.
- **Base64 string**: A raw base64-encoded string (without the `data:` prefix). The image type will be auto-detected from the binary content.
- **Binary object**: An object with a `bytes` field containing a `Uint8Array` of image data, and an optional `path` field with the file name (used for type detection).

Supported image formats: **PNG**, **JPEG**, **TIFF**, **BMP**, **GIF**. The format must also be supported by the browser for display.

## Audio Data

Embedding Atlas can play audio in tooltips and the instances view. Audio column values can be provided in the following formats:

- **URL**: A string starting with `http://` or `https://` pointing to the audio file.
- **Data URL**: A string starting with `data:audio/...` containing inline audio data.
- **Base64 string**: A raw base64-encoded string (without the `data:` prefix). The audio type will be auto-detected from the binary content.
- **Binary object**: An object with a `bytes` field containing a `Uint8Array` of audio data, and an optional `path` field with the file name (used for type detection via file extension).

Supported audio formats: **MP3**, **WAV**, **OGG**, **FLAC**, **AAC**, **M4A (MP4)**, **WebM**. The format must also be supported by the browser for playback.
