import logging
from pathlib import Path
import runpy

from train_config import build_train_config

def main():
    logging.basicConfig(level=logging.INFO)

    train_config = build_train_config()

    openpi_dir = Path(__file__).resolve().parent.parent/"openpi"
    train_script = openpi_dir/ "scripts" / "train.py"

    if not train_script.is_file():
        raise FileNotFoundError(f"Train script not found: {train_script}")

    train_module = runpy.run_path(str(train_script))
    train_module["main"](train_config)


if __name__ == "__main__":
    main()