import shutil
from pathlib import Path

import yaml


def edit_config(
    n_epochs: int | None = None,
    max_len: int | None = None,
    globals_config: str = "../configs/eir-transformer/globals.yaml",
    output_config: str = "../configs/eir-transformer/output_sequence.yaml",
) -> None:
    if n_epochs is not None:
        path = Path(globals_config)
        config = yaml.safe_load(path.read_text())
        config["basic_experiment"]["n_epochs"] = n_epochs
        path.write_text(yaml.safe_dump(config, sort_keys=False, default_flow_style=False))

    if max_len is not None:
        path = Path(output_config)
        config = yaml.safe_load(path.read_text())
        config["output_type_info"]["max_length"] = max_len
        path.write_text(yaml.safe_dump(config, sort_keys=False, default_flow_style=False))


def restore_pretrained(
    pretrained_root: str = "../pretrained/eir-transformer",
    data_folder: str = "../reference_data/eir-transformer",
    runs_folder: str = "../runs",
) -> None:
    pretrained_root = Path(pretrained_root)
    data_folder = Path(data_folder)
    runs_folder = Path(runs_folder)

    if not pretrained_root.is_dir():
        raise FileNotFoundError(
            f"{pretrained_root} not found. Are you running from notebooks/?"
        )

    data_folder.mkdir(parents=True, exist_ok=True)
    for path in (pretrained_root / "data").iterdir():
        shutil.copy2(path, data_folder / path.name)

    for model_folder in (pretrained_root / "models").iterdir():
        target = runs_folder / model_folder.name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(model_folder, target)
