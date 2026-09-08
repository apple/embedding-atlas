"""CLI entry point: uv run python -m benchmark {umap,nndescent}"""

import click


@click.group()
def cli():
    """UMAP / NNDescent benchmark suite."""
    pass


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
