from train_config import build_train_config
from pathlib import Path
import runpy
from openpi.shared import normalize
import numpy as np
from tqdm import tqdm

def main():
    config = build_train_config()
    data_config = config.data.create(config.assets_dirs, config.model)
    openpi_dir = Path(__file__).resolve().parent.parent / "openpi"
    stats_script = openpi_dir / "scripts" / "compute_norm_stats.py"
    stats_module = runpy.run_path(str(stats_script))

    data_loader, num_batches = stats_module["create_torch_dataloader"](
        data_config = data_config,
        action_horizon = config.model.action_horizon,
        batch_size = 32,
        model_config = config.model,
        num_workers = 0,
        )

    running_stats = {
        "state": normalize.RunningStats(),
        "actions": normalize.RunningStats(),
    }

    for batch in tqdm(data_loader, total=num_batches, desc = "计算统计量"):
        for key, accumulator in running_stats.items():
            accumulator.update(np.asarray(batch[key]))

    norm_stats = {
        key:accumulator.get_statistics() 
        for key, accumulator in running_stats.items()
    }
    output_dir = config.assets_dirs / data_config.repo_id
    normalize.save(output_dir, norm_stats)
    print(f"统计量已保存到：{output_dir / 'norm_stats.json'}")

if __name__ == "__main__":
    main()
