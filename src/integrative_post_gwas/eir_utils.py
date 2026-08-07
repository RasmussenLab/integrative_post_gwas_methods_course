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
