from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

LABS = [
    "HDL_cholesterol",
    "Body_mass_index_BMI_IMPED",
    "Total_Triglycerides",
    "Systolic_blood_pressure_automated_reading",
]
DX_CODES = ["DX_E11", "DX_I10", "DX_I25"]

BIN_GRANULARITY = 2
AGE_START_RANGE = (45, 70)
VISIT_RANGE = (3, 9)
VISIT_GAP = 2

LEVEL_RANGE = (20, 80)
DRIFT_SD = 6

BP_DX_THRESHOLD = 75
BP_DX_P = 0.5
RANDOM_DX_P = 0.05


@dataclass
class DemoDFs:
    df_train: pd.DataFrame
    df_test: pd.DataFrame
    valid_ids: pd.Series


def lab_token(trait_name: str, pct: float) -> str:
    bin_upper = int(round(pct / BIN_GRANULARITY)) * BIN_GRANULARITY
    return f"LAB_{trait_name}_{bin_upper}%"


def diagnosis_token(bp_pct: float, rng: np.random.Generator) -> str | None:
    if bp_pct > BP_DX_THRESHOLD and rng.random() < BP_DX_P:
        return "DX_I10"

    if rng.random() < RANDOM_DX_P:
        return str(rng.choice(DX_CODES))

    return None


def build_sequence(patient_id: str, rng: np.random.Generator) -> dict:
    sex = rng.integers(1, 3)
    age = rng.integers(*AGE_START_RANGE)

    level = rng.uniform(*LEVEL_RANGE, size=len(LABS))
    drift = rng.normal(0, DRIFT_SD, size=len(LABS))

    cur_ehr = [f"SEX_{sex}"]
    for n in range(rng.integers(*VISIT_RANGE)):
        cur_ehr.append("[VISIT_START]")
        cur_ehr.append(f"AGE_{age + VISIT_GAP * n}")

        pct = np.clip(level + drift * n, 2, 100)
        for trait_name, trait_pct in zip(LABS, pct):
            cur_ehr.append(lab_token(trait_name=trait_name, pct=trait_pct))

        dx = diagnosis_token(bp_pct=pct[3], rng=rng)
        if dx is not None:
            cur_ehr.append(dx)

        cur_ehr.append("[VISIT_END]")
        cur_ehr.append("[SEP]")

    return {"ID": patient_id, "Sequence": " ".join(cur_ehr)}


def build_sequences_df(
    n_patients: int, prefix: str, rng: np.random.Generator
) -> pd.DataFrame:
    sequences = [
        build_sequence(patient_id=f"{prefix}{i:04d}", rng=rng)
        for i in range(n_patients)
    ]
    return pd.DataFrame(sequences)


def save_validation_ids(valid_ids: pd.Series, output_folder: Path) -> None:
    with open(output_folder / "validation_ids_demo.txt", "w") as handle:
        for sample_id in valid_ids:
            handle.write(f"{sample_id}\n")


def main(
    output_folder: Path = Path("../reference_data/eir-intro-demo"),
    n_train: int = 1000,
    n_test: int = 200,
    valid_fraction: float = 0.15,
    seed: int = 0,
) -> DemoDFs:
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    df_train = build_sequences_df(n_patients=n_train, prefix="TRAIN_", rng=rng)
    df_test = build_sequences_df(n_patients=n_test, prefix="TEST_", rng=rng)

    df_train.to_csv(output_folder / "df_ehr_demo.csv", index=False)
    df_test.to_csv(output_folder / "df_ehr_demo_test.csv", index=False)

    valid_ids = df_train["ID"].sample(frac=valid_fraction, random_state=seed)
    save_validation_ids(valid_ids=valid_ids, output_folder=output_folder)

    return DemoDFs(df_train=df_train, df_test=df_test, valid_ids=valid_ids)


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    main(output_folder=repo_root / "reference_data/eir-intro-demo")
