from pathlib import Path
import re
import glob
import shutil
import subprocess

import yaml
import matplotlib.pyplot as plt
import pandas as pd

from typing import Optional


def parse_sequence(sequence: str) -> pd.DataFrame:
    rows = []
    for chunk in sequence.split("[VISIT_START]")[1:]:
        chunk = chunk.split("[VISIT_END]")[0]
        row, dx = {}, []
        for tok in chunk.split():
            if tok.startswith("AGE_"):
                row["age"] = int(tok[len("AGE_") :])
            elif tok.startswith("LAB_"):
                trait, level = tok[len("LAB_") :].rstrip("%").rsplit("_", 1)
                row[trait] = int(level)
            elif tok.startswith("DX_"):
                dx.append(tok[len("DX_") :])
        row["dx"] = dx
        rows.append(row)
    return pd.DataFrame(rows)


def onset_age(parsed: pd.DataFrame, disease: str) -> int | None:
    hits = parsed[parsed["dx"].apply(lambda d: disease in d)]
    return int(hits["age"].iloc[0]) if len(hits) else None


def plot_real_vs_generated(
    real_seq: str,
    generated_seq: Optional[str],
    lab: str,
    disease: str | None = None,
    prompt_str: str | None = None,
) -> None:
    real = parse_sequence(sequence=real_seq)

    gen = None
    if generated_seq is not None:
        gen = parse_sequence(sequence=generated_seq)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(real["age"], real[lab], marker="o", color="#1f77b4", label="Real")

    if prompt_str is not None:
        cutoff_age = parse_sequence(sequence=prompt_str)["age"].max()
        ax.axvspan(
            ax.get_xlim()[0],
            cutoff_age,
            color="0.85",
            alpha=0.5,
            zorder=0,
            label="Prompt (given)",
        )

    if gen is not None:
        ax.plot(
            gen["age"],
            gen[lab],
            marker="s",
            ls="--",
            color="#d62728",
            label="Generated",
        )

    if disease is not None:
        seq_iter = [(real, "#1f77b4")]
        if gen is not None:
            seq_iter.append((gen, "#d62728"))

        for parsed, color in seq_iter:
            age = onset_age(parsed, disease)
            if age is not None:
                ax.axvline(age, color=color, ls=":", alpha=0.7)
                ax.annotate(
                    f"{disease} onset",
                    xy=(age, ax.get_ylim()[1]),
                    color=color,
                    fontsize=8,
                    rotation=90,
                    va="top",
                    ha="right",
                )

    title = f"Real vs generated trajectory: {lab}"
    if generated_seq is None:
        title = f"Real trajectory: {lab}"

    ax.set_xlabel("Age")
    ax.set_ylabel(f"{lab} (percentile bin)")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.2)
    plt.show()


def aggregate_replicates(reps: list[pd.DataFrame], lab: str) -> pd.DataFrame | None:
    frames = []
    for df in reps:
        if lab not in df.columns:
            continue
        t = df[["age", lab]].reset_index(drop=True)
        t["visit_idx"] = t.index
        frames.append(t)
    if not frames:
        return None
    stacked = pd.concat(frames, ignore_index=True)
    g = stacked.groupby("visit_idx")
    return pd.DataFrame(
        {
            "age": g["age"].mean(),
            "mean": g[lab].mean(),
            "std": g[lab].std(),
            "n": g[lab].size(),
        }
    )


def plot_trajectory_comparison(
    real_seq: str,
    generated: dict[str, str | list[str]],
    labs: list[str],
) -> None:
    parsed_real = parse_sequence(sequence=real_seq)
    parsed_gen = {
        label: [
            parse_sequence(sequence=s)
            for s in (seqs if isinstance(seqs, list) else [seqs])
        ]
        for label, seqs in generated.items()
    }

    ncols = 2
    nrows = (len(labs) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False
    )
    axes_flat = axes.ravel()

    colors = ["#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]

    for ax, lab in zip(axes_flat, labs):
        ax.plot(
            parsed_real["age"],
            parsed_real[lab],
            color="0.6",
            lw=3,
            marker="o",
            label="Real",
        )
        for (label, reps), color in zip(parsed_gen.items(), colors):
            agg = aggregate_replicates(reps=reps, lab=lab)
            if agg is None:
                continue
            se = (agg["std"] / agg["n"] ** 0.5).fillna(0)
            ax.plot(
                agg["age"], agg["mean"], color=color, ls="--", marker="s", label=label
            )
            ax.fill_between(
                agg["age"], agg["mean"] - se, agg["mean"] + se, color=color, alpha=0.2
            )
        ax.set_xlabel("Age")
        ax.set_ylabel(f"{lab} (percentile bin)")
        ax.set_title(lab)
        ax.legend()
        ax.grid(alpha=0.2)

    for ax in axes_flat[len(labs) :]:
        ax.set_visible(False)

    fig.tight_layout()
    plt.show()


def set_manual_inputs(
    prompts: str | list[str],
    config_path: str = "../configs/eir-transformer/output_sequence_test.yaml",
    tabular: Optional[dict | list[dict]] = None,
) -> None:
    if isinstance(prompts, str):
        prompts = [prompts]

    if tabular is not None:
        if isinstance(tabular, dict):
            tabular = [tabular]
        if len(tabular) != len(prompts):
            raise ValueError(f"{len(prompts)} prompts vs {len(tabular)} tabular rows.")

    config_path = Path(config_path)
    with open(config_path) as handle:
        config = yaml.safe_load(handle)

    manual_inputs = []
    for i, p in enumerate(prompts):
        entry = {"ehr": p}
        if tabular is not None:
            entry["biomarker_inputs"] = tabular[i]

        manual_inputs.append(entry)

    config["sampling_config"]["manual_inputs"] = manual_inputs

    with open(config_path, "w") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, default_flow_style=False)


def load_test_sample(
    index: int = 0,
    path: str = "../reference_data/eir-transformer/df_ehr_test.csv",
) -> str:
    df = pd.read_csv(path, index_col="ID")
    return df["Sequence"].iloc[index]


def load_trait_csv(path: Path) -> pd.Series:
    df = pd.read_csv(path).set_index("ID")
    trait = path.stem
    prob1 = [c for c in df.columns if c.endswith("Ensemble Prob 1")]
    if prob1:
        s = df[prob1[0]].rename(trait)
        s.attrs["kind"] = "binary"
        return s
    ensemble = [
        c for c in df.columns if re.fullmatch(rf"{re.escape(trait)} Ensemble", c)
    ]
    if not ensemble:
        raise ValueError(f"No ensemble column found in {path.name}: {list(df.columns)}")
    s = df[ensemble[0]].rename(trait)
    s.attrs["kind"] = "continuous"
    return s


def load_grs_predictions(results_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    paths = sorted(results_dir.glob("*.csv"))
    if not paths:
        raise ValueError(f"No prediction CSVs found in {results_dir}")

    series_list, kinds = [], {}
    for p in paths:
        s = load_trait_csv(p)
        series_list.append(s)
        kinds[s.name] = s.attrs["kind"]

    df_grs = pd.concat(series_list, axis=1)
    df_grs.index = df_grs.index.astype(str)
    trait_kind = pd.Series(kinds, name="kind")
    return df_grs, trait_kind


def match_grs_to_sequences(df_seq: pd.DataFrame, df_grs: pd.DataFrame) -> pd.DataFrame:
    base_id = df_seq.index.str.split("_").str[0]
    df_matched = df_grs.reindex(base_id)
    df_matched.index = df_seq.index
    return df_matched


def plot_validation_loss(
    runs: dict[str, str],
    metric: str = "loss-average",
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, run_folder in runs.items():
        hist = pd.read_csv(Path(run_folder) / "validation_average_history.log")
        ax.plot(hist["iteration"], hist[metric], marker="o", label=label)

    ax.set_xlabel("Iteration")
    ax.set_ylabel(metric)
    ax.set_title("Validation loss")
    ax.legend()
    ax.grid(alpha=0.2)
    plt.show()


def write_grs_tabular_config(
    input_source: str,
    out_path: str,
    columns: list[str],
    modality_dropout_rate: float = 0.0,
) -> None:
    config = {
        "input_info": {
            "input_source": input_source,
            "input_name": "biomarker_inputs",
            "input_type": "tabular",
        },
        "input_type_info": {
            "input_con_columns": columns,
            "modality_dropout_rate": modality_dropout_rate,
        },
        "model_config": {"model_type": "tabular"},
    }
    with open(out_path, "w") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, default_flow_style=False)


def load_test_grs(
    index: int,
    columns: list[str],
    grs_path: str = "../reference_data/eir-transformer/df_grs_test.csv",
    ehr_path: str = "../reference_data/eir-transformer/df_ehr_test.csv",
) -> dict:
    sample_id = pd.read_csv(ehr_path, index_col="ID").index[index]
    df_grs = pd.read_csv(grs_path, index_col="ID")
    return df_grs.loc[sample_id, columns].to_dict()


def read_generated_sequences(output_folder: str) -> list[str]:
    pattern = f"{output_folder}/results/ehr/ehr/samples/*/manual/*_generated.txt"
    hits = glob.glob(pattern)
    if not hits:
        raise ValueError(f"No generated files for {pattern}")
    hits.sort(key=lambda p: int(Path(p).stem.split("_")[0]))
    return [Path(h).read_text() for h in hits]


def read_generated_sequence(output_folder: str) -> str:
    seqs = read_generated_sequences(output_folder=output_folder)
    if len(seqs) != 1:
        raise ValueError(f"Expected one generated sequence, found {len(seqs)}")
    return seqs[0]


def generate_from_prompt(
    prompt: str,
    globals_config: str,
    output_config: str,
    model_glob: str,
    output_folder: str,
    input_config: str | None = None,
    tabular: dict | None = None,
    replicates: int = 1,
) -> str | list[str]:
    prompts = prompt if replicates == 1 else [prompt] * replicates
    tab = tabular
    if replicates > 1 and tabular is not None:
        tab = [tabular] * replicates
    set_manual_inputs(prompts=prompts, config_path=output_config, tabular=tab)

    matches = sorted(glob.glob(model_glob))
    if len(matches) != 1:
        raise ValueError(
            f"Expected one checkpoint for {model_glob}, found {len(matches)}: {matches}"
        )

    shutil.rmtree(output_folder, ignore_errors=True)

    cmd = [
        "eirpredict",
        "--global_configs",
        globals_config,
        "--output_configs",
        output_config,
        "--model_path",
        matches[0],
        "--output_folder",
        output_folder,
    ]
    if input_config is not None:
        cmd += ["--input_configs", input_config]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError(f"eirpredict failed (exit code {result.returncode})")

    if replicates == 1:
        return read_generated_sequence(output_folder=output_folder)
    return read_generated_sequences(output_folder=output_folder)

    return read_generated_sequence(output_folder=output_folder)
