"""CLI entry point: uv run python -m benchmark {umap,nndescent}"""

import click


@click.group()
def cli():
    """UMAP / NNDescent benchmark suite."""


@cli.command()
@click.option("--skip-generate", is_flag=True, help="Skip data generation.")
@click.option("--skip-python", is_flag=True, help="Skip Python benchmarks.")
@click.option("--python-only", is_flag=True, help="Only run Python benchmarks.")
@click.option("--no-gpu", is_flag=True, help="Disable GPU (enabled by default).")
@click.option("--skip-plot", is_flag=True, help="Skip report generation.")
@click.option(
    "--sizes", type=int, multiple=True, default=None, help="Sample sizes (repeatable)."
)
def umap(skip_generate, skip_python, python_only, no_gpu, skip_plot, sizes):
    """Run UMAP benchmarks."""
    from benchmark.umap.run import run

    run(
        skip_generate=skip_generate,
        skip_python=skip_python,
        python_only=python_only,
        gpu=not no_gpu,
        skip_plot=skip_plot,
        sizes=list(sizes) if sizes else None,
    )


@cli.command()
def datasets():
    """Create the sweep datasets and datasets/index.json without running the sweep."""
    from benchmark.umap.sweep_data import ensure_datasets

    ensure_datasets()


@cli.command()
@click.option("--smoke", is_flag=True, help="Default config only, seed 42.")
@click.option("--dataset", "datasets", multiple=True, help="Only these datasets.")
@click.option("--no-gpu", is_flag=True, help="Skip the Rust GPU variants.")
@click.option("--skip-build", is_flag=True, help="Do not rebuild umap-bench.")
@click.option("--report-only", is_flag=True, help="Only regenerate the report.")
@click.option(
    "--report-points",
    default=1500,
    show_default=True,
    help="Points per embedding in the standalone report.html.",
)
def sweep(smoke, datasets, no_gpu, skip_build, report_only, report_points):
    """Resumable parameter sweep: Rust UMAP versus umap-learn 0.6."""
    if not report_only:
        from benchmark.umap.sweep import run

        run(
            smoke=smoke,
            only_datasets=list(datasets) or None,
            no_gpu=no_gpu,
            skip_build=skip_build,
        )
    from benchmark.umap.sweep_report import write_reports

    write_reports()
    from benchmark.umap.report_html import write_report_html
    from benchmark.umap.sweep_manifest import build_manifest

    write_report_html(build_manifest(), points=report_points)


@cli.command()
@click.option("--skip-generate", is_flag=True, help="Skip data generation.")
@click.option("--skip-python", is_flag=True, help="Skip Python benchmarks.")
@click.option("--python-only", is_flag=True, help="Only run Python benchmarks.")
@click.option("--no-gpu", is_flag=True, help="Disable GPU (enabled by default).")
@click.option(
    "--sizes", type=int, multiple=True, default=None, help="Sample sizes (repeatable)."
)
@click.option(
    "--dims", type=int, multiple=True, default=None, help="Dimensions (repeatable)."
)
def nndescent(skip_generate, skip_python, python_only, no_gpu, sizes, dims):
    """Run NNDescent benchmarks."""
    from benchmark.nndescent.run import run

    run(
        skip_generate=skip_generate,
        skip_python=skip_python,
        python_only=python_only,
        gpu=not no_gpu,
        sizes=list(sizes) if sizes else None,
        dims=list(dims) if dims else None,
    )


if __name__ == "__main__":
    cli()
