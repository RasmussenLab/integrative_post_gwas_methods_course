from pathlib import Path

import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def load_latents(latent_dir: Path) -> dict[str, np.ndarray]:
    files = sorted(Path(latent_dir).glob("batch_*.npy"))
    if not files:
        raise ValueError(f"No batch_*.npy found in {latent_dir}")
    records = np.concatenate([np.load(f, allow_pickle=True) for f in files])
    return {r["ID"]: r["Latent"] for r in records}


def to_patient_matrix(
    latents: dict[str, np.ndarray],
    sequences: dict[str, str],
) -> pd.DataFrame:
    ids = sorted(latents.keys() & sequences.keys())
    if not ids:
        raise ValueError("No overlap between latent IDs and sequence IDs.")

    rows = []
    for i in ids:
        latent = latents[i]
        n = min(len(sequences[i].split()), latent.shape[0])
        rows.append(latent[:n].mean(axis=0))

    patient = [i.split("_")[0] for i in ids]
    return pd.DataFrame(np.stack(rows), index=patient).groupby(level=0).mean()


def load_grs(grs_paths: list[Path], patient_ids) -> pd.DataFrame:
    grs = pd.concat([pd.read_csv(p, index_col="ID") for p in grs_paths])
    grs.index = grs.index.astype(str).str.split("_").str[0]
    return grs.groupby(level=0).first().reindex(patient_ids)


def umap_embedding(
    emb: pd.DataFrame,
    n_neighbors: int = 25,
    min_dist: float = 0.15,
    seed: int = 42,
) -> np.ndarray:
    X = StandardScaler().fit_transform(emb.values)
    reducer = umap.UMAP(
        n_neighbors=n_neighbors, min_dist=min_dist, random_state=seed, n_jobs=1
    )
    return reducer.fit_transform(X)


def plot_umap_grid(
    panels: dict[str, tuple[np.ndarray, pd.DataFrame]],
    traits: list[str],
    drivers: list[str] | None = None,
    style: str = "scatter",
    gridsize: int = 14,
) -> None:
    drivers = set(drivers or [])
    models = list(panels)
    fig, axes = plt.subplots(
        len(models),
        len(traits),
        figsize=(3.1 * len(traits), 3.1 * len(models)),
        squeeze=False,
    )

    for row, model in enumerate(models):
        coords, grs = panels[model]
        for col, trait in enumerate(traits):
            ax = axes[row][col]
            v = grs[trait].values
            lo, hi = np.percentile(v, [5, 95])
            if style == "hexbin":
                ax.hexbin(
                    coords[:, 0],
                    coords[:, 1],
                    C=v,
                    reduce_C_function=np.mean,
                    gridsize=gridsize,
                    cmap="viridis",
                    vmin=lo,
                    vmax=hi,
                )
            else:
                ax.scatter(coords[:, 0], coords[:, 1], c=v, cmap="viridis", vmin=lo, vmax=hi, s=14)
            ax.set_xticks([])
            ax.set_yticks([])
            if drivers:
                hue = "#111827" if trait in drivers else "#c44e52"
                for spine in ax.spines.values():
                    spine.set_color(hue)
                    spine.set_linewidth(1.6)
            if row == 0:
                ax.set_title(trait.replace("_", " "), fontsize=8)
            if col == 0:
                ax.set_ylabel(model, fontsize=9)

    fig.tight_layout()
    plt.show()
