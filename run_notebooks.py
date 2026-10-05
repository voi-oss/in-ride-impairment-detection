import logging
import os
import uuid
from pathlib import Path
from typing import Any

import fire
import papermill as pm
from jupyter_client.manager import KernelManager
from jupyter_core.paths import jupyter_runtime_dir


class IPCKernelManager(KernelManager):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kernel_id = str(uuid.uuid4())
        os.makedirs(jupyter_runtime_dir(), exist_ok=True)
        connection_file = os.path.join(
            jupyter_runtime_dir(), f"kernel-{kernel_id}.json"
        )
        super().__init__(
            *args,
            transport="ipc",
            kernel_id=kernel_id,
            connection_file=connection_file,
            **kwargs,
        )


logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def run_notebook(
    notebook_path: Path,
    output_dir: Path,
    parameters: dict,
    save_per_ride: bool = False,
) -> None:
    """Run a single notebook with the given parameters using papermill.

    Args:
        notebook_path: Path to the input notebook.
        output_dir: Directory where the output notebook will be saved.
        parameters: Parameters injected into the notebook's `parameters` cell.
        save_per_ride: If True, save per-ride outcomes to a CSV next to the notebook.
    """
    # glue is always on; per-ride saving is opt-in. Keep both out of the filename.
    suffix = "".join(
        f"_{k}_{v}"
        for k, v in sorted(parameters.items())
        if k not in ("glue", "per_ride_path")
    )
    output_path = output_dir / f"{notebook_path.stem}{suffix}{notebook_path.suffix}"
    if output_path.exists():
        logger.info(
            f"Skipping {notebook_path.name}, output already exists: {output_path}"
        )
        return

    # glue metrics; optionally save per-ride outcomes next to the rendered notebook
    parameters = {
        **parameters,
        "glue": True,
        "per_ride_path": (
            str(output_dir / f"{notebook_path.stem}{suffix}.csv")
            if save_per_ride
            else None
        ),
    }

    try:
        pm.execute_notebook(
            input_path=str(notebook_path),
            output_path=str(output_path),
            parameters=parameters,
            cwd=str(notebook_path.parent),
            progress_bar=False,
            kernel_manager_class="run_notebooks.IPCKernelManager",
        )
    except Exception as e:
        logger.error(f"Failed to execute {notebook_path.name} with {parameters}: {e}")
        raise


def main(
    *notebooks: str,
    output_dir: str = "output",
    log_level: str = "INFO",
    save_per_ride: bool = False,
    **parameters: Any,
) -> None:
    """Run notebooks with the given parameters using papermill.

    Any extra ``--key value`` flags are forwarded as parameters to the
    notebook's ``parameters`` cell (e.g. ``--alpha 0.05 --pe_w 800``). The
    rendered notebooks are saved to ``output_dir`` with the parameter values
    appended to the filename.

    Args:
        notebooks: List of notebook paths to run.
        output_dir: Directory where output notebooks will be saved.
        log_level: Logging level (e.g., "INFO", "DEBUG", "WARNING", "ERROR").
        save_per_ride: If True, save per-ride outcomes to a CSV per notebook.
        parameters: Parameters injected into each notebook's `parameters` cell.
    """
    setup_logging(log_level)

    # absolute so paths injected into notebooks resolve regardless of kernel cwd
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output directory: {output_path.absolute()}")
    logger.info(f"Notebooks to run: {list(notebooks)}")
    logger.info(f"Parameters: {parameters}")

    for notebook in notebooks:
        notebook_path = Path(notebook)
        if not notebook_path.exists():
            logger.error(f"Notebook not found: {notebook_path}")
            continue
        run_notebook(notebook_path, output_path, dict(parameters), save_per_ride)

    logger.info("Notebook execution pipeline completed successfully")


if __name__ == "__main__":
    fire.Fire(main)
