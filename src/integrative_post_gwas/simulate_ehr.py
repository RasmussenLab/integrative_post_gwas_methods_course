from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Callable, Literal

AGE_REF = 60
LAB_AGE_SLOPE = 1.5
LAB_NOISE_SD = 3.0
GAMMA = 1.0

DX_SCALING_FACTOR = 0.2
DX_PERSIST_P = 0.9

BIN_GRANULARITY = 2

AUX_LINKS_DEFAULT = {}


@dataclass
class Trait:
    name: str
    type: Literal["con", "cat"]

    def column_to_load(self):
        if self.type == "con":
            return f"{self.name} Ensemble"
        if self.type == "cat":
            return f"{self.name} Ensemble Prob 1"


def gather_predictions(preds_folder: Path, keep_list: list[Trait]) -> pd.DataFrame:
    dfs = []

    for trait in keep_list:
        target_path = (preds_folder / trait.name).with_suffix(".csv")

        ensemble_col = trait.column_to_load()
        df = pd.read_csv(
            target_path,
            usecols=["ID", ensemble_col],
            index_col="ID",
        )
        df = df.rename(columns={ensemble_col: trait.name})
        dfs.append(df)

    df_all = pd.concat(dfs, axis=1)
    df_all.index = df_all.index.astype(str)

    return df_all


def convert_df_to_percentiles(df: pd.DataFrame) -> pd.DataFrame:
    ranks = df.rank(pct=True)
    return np.ceil(ranks * 100).astype(int)


def lab_token(
    trait_name: str,
    base_pct: int,
    age: int,
    aux_effect: float = 0.0,
) -> str:

    slope = LAB_AGE_SLOPE * (base_pct / 100) ** GAMMA
    delta_age = age - AGE_REF

    level = slope * delta_age + base_pct + aux_effect

    cur_noise = np.random.normal(0, LAB_NOISE_SD)
    level += cur_noise

    level = int(np.clip(level, 1, 100))
    bin_upper = ((level - 1) // BIN_GRANULARITY + 1) * BIN_GRANULARITY
    return f"LAB_{trait_name}_{bin_upper}%"


def diagnosis_token(
    trait_name: str,
    risk_pct: int,
    already_dx: bool,
    aux_effect: float = 0.0,
) -> tuple[str | None, bool]:
    if already_dx:
        token = f"DX_{trait_name}" if np.random.rand() < DX_PERSIST_P else None
        return token, True
    risk_adj = np.clip(risk_pct + aux_effect, 1, 100)
    if np.random.rand() < DX_SCALING_FACTOR * (risk_adj / 100) ** 2:
        return f"DX_{trait_name}", True
    return None, False


def aux_effect_for(
    trait_name: str,
    aux_links: dict[str, dict[str, float]],
    individual_info: dict,
) -> float:
    effect = 0.0
    for aux_name, weight in aux_links.get(trait_name, {}).items():
        pct = individual_info[aux_name]
        effect += weight * (pct - 50) / 50
    return effect


def build_sequence(
    individual_info: dict,
    target_traits: list[Trait],
    aux_links: dict[str, dict[str, float]] | None = None,
    replicates: int = 10,
    min_visits: int = 3,
    max_visits: int = 7,
) -> list[str]:
    if aux_links is None:
        aux_links = {}

    visits_this_individual = []

    if_ = individual_info

    for r in range(replicates):
        n_visits = np.random.randint(min_visits, max_visits + 1)
        diagnosis_track = set()

        age = if_["age"]
        sex = if_["sex"]

        cur_ehr = []
        cur_ehr.append(f"SEX_{sex}")
        for n in range(int(n_visits)):
            cur_ehr.append("[VISIT_START]")

            age += np.random.randint(1, 5)
            cur_ehr.append(f"AGE_{age}")

            for t in target_traits:
                if t.type == "con":
                    cur_token = lab_token(
                        trait_name=t.name,
                        base_pct=if_[t.name],
                        age=age,
                        aux_effect=aux_effect_for(
                            trait_name=t.name,
                            aux_links=aux_links,
                            individual_info=if_,
                        ),
                    )
                elif t.type == "cat":
                    already_diagnosed = t.name in diagnosis_track
                    cur_token, is_positive = diagnosis_token(
                        trait_name=t.name,
                        risk_pct=if_[t.name],
                        already_dx=already_diagnosed,
                        aux_effect=aux_effect_for(
                            trait_name=t.name,
                            aux_links=aux_links,
                            individual_info=if_,
                        ),
                    )

                    if is_positive:
                        diagnosis_track.add(t.name)
                else:
                    raise ValueError()

                if cur_token is not None:
                    cur_ehr.append(cur_token)
            cur_ehr.append("[VISIT_END]")
            if n < n_visits - 1:
                cur_ehr.append("[SEP]")

        ehr_string = " ".join(cur_ehr)
        visits_this_individual.append(ehr_string)

    return visits_this_individual


def build_traits(con_traits: list[str], cat_traits: list[str]) -> list[Trait]:
    all_traits = []

    for con_trait in con_traits:
        all_traits.append(Trait(name=con_trait, type="con"))

    for cat_trait in cat_traits:
        all_traits.append(Trait(name=cat_trait, type="cat"))

    return all_traits


def build_sequences_df(
    df: pd.DataFrame,
    target_traits: list[Trait],
    aux_links: dict[str, dict[str, float]] | None = None,
) -> pd.DataFrame:
    all_sequences = []
    for individual_row in df.itertuples():
        row_dict = individual_row._asdict()

        id_ = row_dict["Index"]

        individual_sequences = build_sequence(
            individual_info=row_dict,
            target_traits=target_traits,
            aux_links=aux_links,
            replicates=10,
            min_visits=3,
            max_visits=7,
        )

        for visit_n, sequence in enumerate(individual_sequences):
            cur_id = f"{id_}_{visit_n}"
            all_sequences.append((cur_id, sequence))

    df_sequences = pd.DataFrame(all_sequences, columns=["ID", "Sequence"])

    return df_sequences


def split_patient_ids(
    patient_ids: list[str],
    test_size: float = 0.2,
    val_size: float = 0.1,
    seed: int = 42,
) -> tuple[list[str], list[str], list[str]]:
    train_val_ids, test_ids = train_test_split(
        patient_ids, test_size=test_size, random_state=seed
    )
    val_fraction = val_size / (1.0 - test_size)
    train_ids, val_ids = train_test_split(
        train_val_ids, test_size=val_fraction, random_state=seed
    )
    return train_ids, val_ids, test_ids


def subset_sequences_by_patient(
    df_sequences: pd.DataFrame, patient_ids: set[str]
) -> pd.DataFrame:
    base_ids = df_sequences["ID"].str.split("_").str[0]
    return df_sequences[base_ids.isin(patient_ids)]


def trim_sequence(sequence: str, stop_when: Callable[[str], bool]) -> str:
    tokens = sequence.split(" ")
    for i, token in enumerate(tokens):
        if stop_when(token):
            return " ".join(tokens[: i + 1])
    raise ValueError(f"No stop token found in sequence: {sequence}")


def trim_sequences_df(
    df_sequences: pd.DataFrame, stop_when: Callable[[str], bool]
) -> pd.DataFrame:
    df = df_sequences.copy()
    df["Sequence"] = df["Sequence"].apply(lambda s: trim_sequence(s, stop_when))
    return df


@dataclass
class SplitDFs:
    df_seq_train: pd.DataFrame
    df_seq_val: pd.DataFrame
    df_seq_test: pd.DataFrame


def split_ids(df_grs_and_real: pd.DataFrame, df_sequences: pd.DataFrame) -> SplitDFs:
    train_ids, val_ids, test_ids = split_patient_ids(
        patient_ids=list(df_grs_and_real.index)
    )

    df_seq_train = subset_sequences_by_patient(
        df_sequences=df_sequences, patient_ids=set(train_ids) | set(val_ids)
    )
    df_seq_test = subset_sequences_by_patient(
        df_sequences=df_sequences, patient_ids=set(test_ids)
    )
    df_seq_val = subset_sequences_by_patient(
        df_sequences=df_sequences, patient_ids=set(val_ids)
    )

    return SplitDFs(
        df_seq_train=df_seq_train, df_seq_val=df_seq_val, df_seq_test=df_seq_test
    )


def save_validation_ids(df_seq_val: pd.DataFrame, output_folder: Path) -> None:
    with open(output_folder / "validation_ids.txt", "w") as handle:
        for sample_id in df_seq_val["ID"]:
            handle.write(f"{sample_id}\n")


def _verify_traits(target_folder: Path, target_traits: list[Trait]):
    all_traits = {i.stem for i in target_folder.iterdir()}

    for trait in target_traits:
        trait_name = trait.name
        if trait_name not in all_traits:
            raise ValueError(f"Trait {trait} not found in target traits.")


def main(
    input_root: Path,
    output_folder: Path,
    con_traits: list[str],
    cat_traits: list[str],
    aux_links: dict[str, dict[str, float]] | None = None,
):
    if aux_links is None:
        aux_links = AUX_LINKS_DEFAULT

    for driven_trait in aux_links:
        if driven_trait not in con_traits and driven_trait not in cat_traits:
            raise ValueError(
                f"aux_links trait {driven_trait!r} is not among con_traits or cat_traits"
            )

    target_traits = build_traits(con_traits=con_traits, cat_traits=cat_traits)

    target_folder = input_root / "output/results/"
    _verify_traits(target_folder=target_folder, target_traits=target_traits)

    df_preds = gather_predictions(preds_folder=target_folder, keep_list=target_traits)
    df_percentiles = convert_df_to_percentiles(df=df_preds)
    df_labels = pd.read_csv(
        input_root / "reference_data/penncath_hg38/penncath.csv",
        index_col="ID",
        usecols=["ID", "age", "sex"],
    )
    df_labels.index = df_labels.index.astype(str)
    df_joined = pd.merge(left=df_percentiles, right=df_labels, on="ID")

    aux_biomarkers = sorted({aux for links in aux_links.values() for aux in links})
    if aux_biomarkers:
        aux_traits = build_traits(con_traits=aux_biomarkers, cat_traits=[])
        _verify_traits(target_folder=target_folder, target_traits=aux_traits)
        df_aux = gather_predictions(preds_folder=target_folder, keep_list=aux_traits)
        df_aux_pct = convert_df_to_percentiles(df=df_aux)
        df_joined = df_joined.join(df_aux_pct)

    df_sequences = build_sequences_df(
        df=df_joined,
        target_traits=target_traits,
        aux_links=aux_links,
    )

    split_dfs = split_ids(df_grs_and_real=df_joined, df_sequences=df_sequences)

    output_folder.mkdir(parents=True, exist_ok=True)
    save_validation_ids(df_seq_val=split_dfs.df_seq_val, output_folder=output_folder)

    df_test_first_visit = trim_sequences_df(
        df_sequences=split_dfs.df_seq_test, stop_when=lambda t: t == "[VISIT_END]"
    )
    df_test_visit_start = trim_sequences_df(
        df_sequences=split_dfs.df_seq_test, stop_when=lambda t: t.startswith("AGE_")
    )

    split_dfs.df_seq_train.to_csv(output_folder / "df_ehr.csv", index=False)
    split_dfs.df_seq_test.to_csv(output_folder / "df_ehr_test.csv", index=False)
    df_test_first_visit.to_csv(
        output_folder / "df_ehr_test_first_visit.csv", index=False
    )
    df_test_visit_start.to_csv(
        output_folder / "df_ehr_test_visit_start.csv", index=False
    )


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    main(
        input_root=repo_root,
        output_folder=repo_root / "reference_data/eir-transformer",
    )
