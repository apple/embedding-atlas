# Copyright (c) 2025 Apple Inc. Licensed under MIT License.

"""Regression tests for embedding-atlas CLI option handling."""

from click.testing import CliRunner

from embedding_atlas.cli import main


def test_z_requires_x_and_y():
    """--z names a pre-computed column, so it must not be accepted on its own (it
    would otherwise be passed to projection generation, auto-enabling 3D and being
    overwritten by the generated projection_z)."""
    runner = CliRunner()

    # --z alone is rejected before any data is loaded.
    result = runner.invoke(main, ["dummy.csv", "--z", "depth"])
    assert result.exit_code != 0
    assert "--z" in result.output
    assert "umap-n-components" in result.output

    # --z with only --x is still rejected.
    result = runner.invoke(main, ["dummy.csv", "--x", "px", "--z", "depth"])
    assert result.exit_code != 0
    assert "requires both --x and --y" in result.output


def test_z_with_x_and_y_passes_validation():
    """--z together with --x and --y is a valid pre-computed 3D input; it must pass
    the option guard (failing later only because the dummy dataset cannot load)."""
    runner = CliRunner()
    result = runner.invoke(main, ["dummy.csv", "--x", "px", "--y", "py", "--z", "pz"])
    # It should NOT fail on the --z guard (which is a click.UsageError mentioning z).
    assert "requires both --x and --y" not in result.output
