from pathlib import Path
import re

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
                row["age"] = int(tok[len("AGE_"):])
            elif tok.startswith("LAB_"):
                trait, level = tok[len("LAB_"):].rstrip("%").rsplit("_", 1)
                row[trait] = int(level)
            elif tok.startswith("DX_"):
                dx.append(tok[len("DX_"):])
        row["dx"] = dx
        rows.append(row)
    return pd.DataFrame(rows)


def onset_age(parsed: pd.DataFrame, disease: str) -> int | None:
    hits = parsed[parsed["dx"].apply(lambda d: disease in d)]
    return int(hits["age"].iloc[0]) if len(hits) else None


def plot_real_vs_generated(
    real_seq: str, generated_seq: Optional[str], lab: str, disease: str | None = None,
) -> None:
    real = parse_sequence(sequence=real_seq)

    gen = None
    if generated_seq is not None:
        gen = parse_sequence(sequence=generated_seq)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(real["age"], real[lab], marker="o", color="#1f77b4", label="Real")

    if gen is not None:
        ax.plot(
            gen["age"], gen[lab], marker="s", ls="--", color="#d62728", label="Generated",
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
                    f"{disease} onset", xy=(age, ax.get_ylim()[1]),
                    color=color, fontsize=8, rotation=90, va="top", ha="right",
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


def load_trait_csv(path: Path) -> pd.Series:
  df = pd.read_csv(path).set_index("ID")
  trait = path.stem
  prob1 = [c for c in df.columns if c.endswith("Ensemble Prob 1")]
  if prob1:
      s = df[prob1[0]].rename(trait)
      s.attrs["kind"] = "binary"
      return s
  ensemble = [c for c in df.columns if re.fullmatch(rf"{re.escape(trait)} Ensemble", c)]
  if not ensemble:
      raise ValueError(f"No ensemble column found in {path.name}: {list(df.columns)}")
  s = df[ensemble[0]].rename(trait)
  s.attrs["kind"] = "continuous"
  return s