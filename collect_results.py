import logging
from pathlib import Path

import fire
import pandas as pd
import scrapbook as sb
from tqdm import tqdm

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


def collect_glue_values(output_dir: Path) -> pd.DataFrame:
    """Collect glue values from output notebooks into a DataFrame.

    Args:
        output_dir: Directory containing output notebooks

    Returns:
        DataFrame with one row per notebook and one column per glue key
    """
    rows = []

    # Find all notebook files
    for notebook_file in tqdm(sorted(output_dir.glob("*.ipynb"))):
        # Read the notebook using scrapbook
        nb = sb.read_notebook(str(notebook_file))

        # Extract all glued data
        row = {"notebook": notebook_file.stem}
        for key, scrap in nb.scraps.items():
            row[key] = scrap.data
        rows.append(row)

    return pd.DataFrame(rows)


def main(
    output_dir: str = "output",
    log_level: str = "INFO",
) -> None:
    """
    Extract glue values from notebooks and save them to a CSV.

    Args:
        output_dir: Directory containing output notebooks
        log_level: Logging level (e.g., "INFO", "DEBUG", "WARNING", "ERROR")
    """
    setup_logging(log_level)
    output_path = Path(output_dir)

    logger.info("Extracting glue values from output notebooks")
    df = collect_glue_values(output_path)

    csv_filename = output_path / "results.csv"
    df.to_csv(csv_filename, index=False, float_format="%.4g")
    logger.info(f"Saved {len(df)} notebooks -> {csv_filename.name}")


if __name__ == "__main__":
    fire.Fire(main)
