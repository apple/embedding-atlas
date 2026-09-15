# Mobile (iOS / Android) — status and plan

The desktop app uses a **PyInstaller Python sidecar** speaking HTTP to a
Svelte+WebGPU frontend. That architecture doesn't translate to mobile:
iOS doesn't ship a Python runtime, Android doesn't let us spawn long-lived
native subprocesses from a user app sandbox, and shipping a full Python
interpreter plus duckdb/pyarrow/pandas would hit both binary-size and
App Store review limits.

For mobile we ship the **frontend-only** build instead:

- **DuckDB-WASM** in the webview handles all queries in-process.
- The `FileViewer` (already used by the static web export) lets users
  drop or pick a file; the file is mapped into DuckDB-WASM via
  `registerFileBuffer`.
- No sidecar, no spawn, no network server — just the webview + WASM.

This is already working today in Safari / Chrome on iOS / Android. The
remaining work for Tauri-wrapped mobile builds is:

1. **Tauri Mobile init**:

   ```bash
   cd apps/desktop/src-tauri
   cargo tauri ios init
   cargo tauri android init
   ```

   (Requires Xcode + Apple Developer account for iOS, Android Studio + NDK
   for Android.)

2. **Conditional sidecar skip** — detect the mobile target at runtime in
   Svelte (`TAURI_PLATFORM` or the Tauri `platform()` API) and navigate
   straight to the `FileViewer` route instead of the home-screen file
   picker that spawns the sidecar.

3. **Signing / store setup**:
   - iOS: Apple Developer Program ($99/year) + provisioning profiles
     configured via Xcode.
   - Android: a signing keystore committed as a GitHub secret.

4. **CI** — `app-mobile-release.yml` (scaffolded in `.github/workflows/`)
   is currently a placeholder with manual-dispatch TODOs. Turn on the
   actual build steps once (1)–(3) are done.

### Why not ship now?

- **Mobile webviews don't yet have WebGPU** on the common shipping
  versions. Chrome on Android and Safari on iOS (as of late 2025) are
  both partially rolled out; the app would fall back to WebGL, which is
  much slower for the 1 M+-point embedding view. The built-in WebGPU
  banner we added already warns users of this.
- **File size for GeoParquet datasets** on mobile storage is tight. We'd
  probably add a progressive-load or remote-query mode in a separate
  iteration.

When ready, uncomment the jobs in `.github/workflows/app-mobile-release.yml`.
