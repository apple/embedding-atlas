"""
Generate assets by copying viewer dist and modifying HTML files.
"""

import hashlib
import shutil
import sys
from pathlib import Path
from textwrap import dedent, indent


def insert_script(html_file_path: Path, snippet: str):
    if html_file_path.exists():
        with open(html_file_path, "r", encoding="utf-8") as f:
            content = f.read()

        snippet = (
            '<script type="module">\n'
            + indent(dedent(snippet), "  ").rstrip().lstrip("\n")
            + "\n</script>"
        )

        modified_content = content.replace(
            "</style>",
            "</style>\n" + indent(snippet, "    "),
            1,
        )

        with open(html_file_path, "w", encoding="utf-8") as f:
            f.write(modified_content)


def file_hash(path: Path):
    hasher = hashlib.sha1()
    with open(path, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()


def main():
    # Get the directory where this script is located
    script_dir = Path(__file__).parent

    # Define paths relative to the script directory
    viewer_dist = script_dir / "../viewer/dist"
    public_dir = script_dir / "public"

    # Ensure viewer dist exists
    if not viewer_dist.exists():
        print(f"Error: {viewer_dist} does not exist", file=sys.stderr)
        sys.exit(1)

    # Create the /app page
    print("Creating /app page...")
    app_dir = public_dir / "app"

    # Remove existing app directory if it exists
    if app_dir.exists():
        shutil.rmtree(app_dir)

    # Copy viewer dist to public/app
    shutil.copytree(viewer_dist, app_dir)

    # Modify the index.html file
    insert_script(
        app_dir / "index.html",
        """
        window.EMBEDDING_ATLAS_CONFIG = { "home": "file-viewer" };
        """,
    )

    # Create the /examples/app page
    print("Creating /examples/app page...")
    examples_dir = public_dir / "examples"
    examples_app_dir = examples_dir / "app"

    # Remove existing examples/app directory if it exists
    if examples_dir.exists():
        shutil.rmtree(examples_dir)

    # Create examples directory if it doesn't exist
    examples_dir.mkdir(parents=True, exist_ok=True)

    # Copy viewer dist to public/examples/app
    shutil.copytree(viewer_dist, examples_app_dir)

    # Copy datasets.js to assets folder (with a random hash suffix)
    datasets_js = f"datasets-{file_hash(script_dir / 'examples/datasets.js')[:8]}.js"
    shutil.copyfile(
        script_dir / "examples/datasets.js",
        examples_app_dir / "assets" / datasets_js,
    )
    insert_script(
        examples_app_dir / "index.html",
        """
        import { loadDataFromUrl } from "./assets/__datasets_js__";
        window.EMBEDDING_ATLAS_CONFIG = { home: "file-viewer", loadDataFromUrl: loadDataFromUrl };
        """.replace("__datasets_js__", datasets_js),
    )

    if (script_dir / ".cache").exists():
        shutil.copytree(
            script_dir / ".cache", examples_dir / "cache", dirs_exist_ok=True
        )

    print("Asset generation completed successfully!")


if __name__ == "__main__":
    main()
