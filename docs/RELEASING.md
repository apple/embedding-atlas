# Releasing geospatial-atlas

Three release streams. The desktop app uses bare `v*` tags (so Zenodo's
GitHub integration archives clean version strings like `v0.0.7`); the
Python package and static web viewer keep prefixed tags to avoid
colliding with the repo-wide desktop scheme.

| Stream            | Tag prefix | Distro                                          | Workflow                                                |
| ----------------- | ---------- | ----------------------------------------------- | ------------------------------------------------------- |
| Desktop app       | `v*`       | `.dmg` / `.AppImage` / `.deb` / `.msi` / `.exe` | `.github/workflows/app-release.yml`                     |
| Python package    | `py-v*`    | PyPI wheel (`geospatial-atlas`)                 | (add later — see _Python release_ below)                |
| Static web viewer | `web-v*`   | GitHub Pages                                    | `.github/workflows/deploy-gh-pages.sh` (manual for now) |

Desktop releases up through `app-v0.0.6` used the `app-v*` prefix; the
scheme switched at v0.0.7 when the Zenodo auto-DOI integration went
live.

Mobile is documented in [MOBILE.md](./MOBILE.md) and not shipped yet.

---

## Cutting a desktop release

Minimal path:

```bash
# 1. Bump the version in apps/desktop/package.json ("version").
#    (electron-builder reads that single source of truth — no other
#    files to bump since the Tauri shell was replaced with Electron.)
#
# 2. Update CHANGELOG.md with the highlights.
#
# 3. Commit the version bump.
git commit -am "chore(desktop): bump version to 0.0.7"

# 4. Tag and push.
git tag v0.0.7
git push origin main v0.0.7
```

Pushing the `v0.0.7` tag triggers
`.github/workflows/app-release.yml`, which:

1. Spawns four matrix builds: **macOS arm64**, **macOS x64**, **Linux x64**,
   **Windows x64**.
2. Builds the viewer → backend static → PyInstaller sidecar → Electron
   app (via `electron-builder`) for each.
3. Uploads `.dmg`, `.deb`, `.rpm`, `.msi`, and `.exe` artifacts.
4. Creates a **draft** GitHub Release with the tag as name and all
   artifacts attached.
5. You review the draft on github.com, write release notes (or let the
   auto-generated ones stand), and click **Publish**.

### v0.0.1 quickstart (no signing yet)

For the very first release you can ship **unsigned** builds. Expect
these user-facing caveats:

- **macOS**: First-launch warning: right-click → Open. Or after Sequoia
  (15.1+): System Settings → Privacy & Security → "Open Anyway".
- **Windows**: SmartScreen will show "Windows protected your PC — Don't
  run". Users click _More info → Run anyway_. This goes away once you
  have an EV code-signing cert.
- **Linux**: No prompt. `.AppImage` should be `chmod +x`'d before run.

This is fine for a v0.0.x / pre-1.0 release where early users tolerate
friction. Add signing before v0.1.

---

## Semver

We follow **semver** for each stream independently:

- `0.0.x`: unstable, breakage expected every version.
- `0.y.0`: usable but API may shift; document breakage in changelog.
- `1.0.0`: commit to backwards compatibility for the public API
  (CLI flags, backend HTTP schema, exported viewer's `metadata.json`).

Desktop app version must match the Python package version **after 1.0** so
`embedding-atlas==X.Y.Z` speaks to `Geospatial Atlas vX.Y.Z`. Before 1.0 they
can drift (desktop is `v0.0.7`, Python is `0.20.0`).

---

## Signing & notarization — unlocking v0.1

When you outgrow the unsigned warnings:

### macOS

1. Join the [Apple Developer Program](https://developer.apple.com/programs/)
   ($99/year).
2. Generate a **Developer ID Application** certificate in Xcode or the
   Apple Developer portal.
3. Export the `.p12`, base64-encode it:
   ```bash
   base64 -i cert.p12 -o cert.b64
   ```
4. Add these to repo **Settings → Secrets and variables → Actions**
   (electron-builder's standard env vars):
   - `CSC_LINK` = contents of `cert.b64` (or an https URL to it)
   - `CSC_KEY_PASSWORD` = the `.p12` password
   - `APPLE_ID`
   - `APPLE_APP_SPECIFIC_PASSWORD` (from appleid.apple.com)
   - `APPLE_TEAM_ID`

The workflow currently sets `CSC_IDENTITY_AUTO_DISCOVERY=false` to
force unsigned builds; remove that line when you wire secrets up.
electron-builder auto-enables notarization when the `APPLE_*` env
vars are present.

### Windows

EV code-signing certs cost $200–400/year. Then:

- `CSC_LINK` (Windows runner) = base64 of `.pfx`
- `CSC_KEY_PASSWORD` = the `.pfx` password

### Linux

Linux doesn't have a Gatekeeper-equivalent. Just sign `.deb` / `.rpm` with
`dpkg-sig` / `rpm --addsign` if you want repository integration.

---

## Changelog

Keep a top-level `CHANGELOG.md`. Prepend on each release:

```md
## v0.0.7 — 2026-05-01

**Desktop release.**

### Added

- Native macOS / Linux / Windows app (Electron + PyInstaller sidecar).
- Fast path for GeoParquet files via DuckDB spatial (`ST_X`/`ST_Y`).
- Live progress bar during load (DuckDB `query_progress()`).
- Per-dataset URL-hash persistence.
- Drag-and-drop dataset loading.
- Row-limit input for sampling large files.

### Known issues

- Unsigned: first-launch warnings on macOS + Windows.
- iOS / Android not yet shipped — see docs/MOBILE.md.
```

---

## Python release (not yet wired up)

For when you want to release the `embedding-atlas` wheel to PyPI:

```bash
# Bump packages/backend/pyproject.toml version.
git tag py-v0.20.1
git push origin py-v0.20.1
```

A future `py-release.yml` workflow would:

- `uv build --wheel --sdist` in `packages/backend/`.
- Publish via `twine` using `PYPI_TOKEN` secret.

---

## Web (static viewer) release

The static viewer currently deploys via the existing
`scripts/deploy-gh-pages.sh`. For now that's run manually. Tag format
`web-v*` is reserved for a future automated workflow.

---

## Post-release verification

After a release publishes:

1. Download the `.dmg` / `.deb` / `.msi` from the GitHub Release.
2. Install on a clean VM (or at least `rm -rf ~/Library/Application\ Support/io.github.do-me.geospatial-atlas`).
3. Open a Parquet file with `lon`/`lat`.
4. Confirm: WebGPU probe passes, load progresses, view restores on relaunch.

If the release is broken, **delete the GitHub Release + the tag**
(`git tag -d v0.0.7 && git push --delete origin v0.0.7`), fix,
re-tag.
