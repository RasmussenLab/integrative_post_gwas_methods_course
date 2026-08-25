# Integrative Post-GWAS Methods: Advanced Statistics, Functional Genomics, and Machine Learning

Course material and notes for the PhD course "Integrative Post-GWAS Methods: Advanced
Statistics, Functional Genomics, and Machine Learning". This covers Day 3, taught by the
Simon Rasmussen Group at the University of Copenhagen.

## Before you start

We will be using GitHub Codespaces for the sessions today. Please follow the steps in
this guide before we begin:

- [GitHub Codespaces Setup Guide](https://docs.google.com/document/d/1tHPCmgBcmxB7Dms9b3lJ32briWVFrLQ4U2CxUYsmBwE/edit?tab=t.0)

## Day 3 - 26.08.2026

**Machine Learning, PRS, and Risk Prediction**

> **Note**: the timings are a guide, and may shift a bit depending on how the sessions go.

| Time | Topic | Teacher | Notebook | Slides |
|---|---|---|---|---|
| 09:00-09:15 | Recap of Day 2 | Ditte | | |
| 09:15-10:00 | Introduction to machine learning and PRS | Simon W | | [PDF](slides/2026_08_25_intro_to_machine_learning_and_prs.pdf) |
| 10:00-10:20 | Introduction to EIR-FM | Simon R | | |
| 10:20-10:40 | *Break* | | | |
| 10:40-11:40 | Application of EIR-FM | Jiyeon, Simon(s), Magnus | [eir-fm-combined.ipynb](notebooks/eir-fm-combined.ipynb) | [PDF](slides/01_eir-fm-combined_slides.pdf) |
| 11:40-12:00 | Discussion (pros and cons of linear and non-linear methods) | Jiyeon, Simon(s), Magnus | | |
| 12:00-13:00 | *Lunch break* | | | |
| 13:00-13:45 | Introduction of EHR modality, transformers and disease trajectories | Magnus | | |
| 13:45-14:05 | EIR and EIR-transformer | Jiyeon, Simon(s), Magnus | [eir-intro.ipynb](notebooks/eir-intro.ipynb) | [PDF](slides/02_eir-intro_slides.pdf) |
| 14:05-14:25 | *Break* | | | |
| 14:25-15:25 | Application of EIR-Transformer | Jiyeon, Simon(s), Magnus | [eir-transformer.ipynb](notebooks/eir-transformer.ipynb) | [PDF](slides/03_eir-transformer_slides.pdf) |
| 15:25-16:00 | Group activity and discussion (which element from the course can you use in your project) | Simon R | | |

## Teachers

| Name | Email |
|---|---|
| Simon Rasmussen | srasmuss@sund.ku.dk |
| Magnus Tvede Jungersen | magnus.jungersen@sund.ku.dk |
| Jiyeon Min | jiyeon.min@sund.ku.dk |
| Arnór Ingi Sigurdsson | arnor.sigurdsson@sund.ku.dk |
| Simon Wengert | simon.wengert@sund.ku.dk |

## Notebooks

Take these in order:

1. [eir-fm-combined.ipynb](notebooks/eir-fm-combined.ipynb) - polygenic scores, then a
   pre-trained genomic foundation model applied to the PennCATH cohort to predict ~300
   phenotypes (GRS; genomic representation scores) per individual.
2. [eir-intro.ipynb](notebooks/eir-intro.ipynb) - the EIR configuration files and the
   `eirtrain` and `eirpredict` commands.
3. [eir-transformer.ipynb](notebooks/eir-transformer.ipynb) - a GPT style transformer
   trained on simulated EHR trajectories, and whether adding the GRS values from the
   first notebook as an extra input makes it any better.

Slides for each of these are under [slides/](slides/).

## Setup (not needed today)

You only need this if you want to run the notebooks on your own machine after the
course. For the sessions today, follow the Codespaces guide linked above instead.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.zshrc  # or "source ~/.bashrc", reload PATH so uv is available
uv venv
uv sync
uv pip install "git+https://github.com/arnor-sigurdsson/EIR-auto-GP.git@27928f"
uv run jupyter lab
```

Then open e.g. the notebooks under `./notebooks`
