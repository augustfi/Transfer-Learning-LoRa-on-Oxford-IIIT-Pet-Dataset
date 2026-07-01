# Transfer Learning with LoRA — Oxford-IIIT Pet

Transfer-learning experiments on the **Oxford-IIIT Pet** dataset (37 breeds, 12 cat +
25 dog) with a pre-trained **ResNet-34** backbone, plus **Low-Rank Adaptation (LoRA)**
on **ResNet-50**. The code covers binary cat/dog classification, 37-class breed
classification, fine-tuning depth, gradual unfreezing, limited-data / regularization
studies, an imbalanced-class study with a post-hoc confusion analysis, and a full
LoRA vs full-fine-tuning comparison.

The full write-up is in [`DD2424_final_report_group2.pdf`](DD2424_final_report_group2.pdf)
(root). [`Task1.png`](Task1.png) and [`Task2.png`](Task2.png) are the two task
illustrations referenced by the report.

---

## Table of contents
- [Repository layout](#repository-layout)
- [Setup](#setup)
  - [1. Python environment](#1-python-environment)
  - [2. The dataset (auto-download)](#2-the-dataset-auto-download)
  - [3. Hardware](#3-hardware)
- [How the code maps to the report](#how-the-code-maps-to-the-report)
- [Running the experiments](#running-the-experiments)
  - [General rule: run each script from its own folder](#general-rule-run-each-script-from-its-own-folder)
  - [5.1 Binary classification](#51-binary-classification--srcbinary_classification)
  - [5.2 Breed classification](#52-breed-classification--srcbreed_classification)
  - [5.2.1 Fine-tuning L layers](#521-fine-tuning-l-layers--srcfinetune_l_layers)
  - [5.2.2 Gradual unfreezing](#522-gradual-unfreezing--srcgradual_unfreezing)
  - [5.2.3 Limited data & regularization](#523-limited-data--regularization--srclimited_data)
  - [5.2.4 Imbalanced classes + confusion analysis](#524-imbalanced-classes--confusion-analysis--srcimbalanced)
  - [5.3 / 5.4 LoRA experiments](#53--54-lora-experiments--srclora)
- [Where outputs go](#where-outputs-go)
- [Known caveats](#known-caveats)
- [Authors](#authors)

---

## Repository layout

```
.
├── DD2424_final_report_group2.pdf   # the report
├── Task1.png  Task2.png             # task illustrations
├── README.md
├── requirements.txt
├── data/
│   ├── images/                      # raw Oxford-IIIT Pet images (reference copy)
│   ├── annotations/                 # raw list.txt / trainval.txt / test.txt / trimaps
│   └── figures/                     # ALL generated plots, one folder per experiment
│       ├── binary_classification/
│       ├── breed_classification/
│       ├── finetune_l_layers/
│       ├── gradual_unfreezing/
│       ├── limited_data/
│       ├── imbalanced/              # + preds/ (prediction dumps) + per_seed/ (heatmaps)
│       └── lora/                    # + rank_sweep/ , full_vs_lora/ subfolders
└── src/                             # one folder per report experiment
    ├── binary_classification/beginning.py
    ├── breed_classification/multi_class.py
    ├── finetune_l_layers/fine_tune_l_layers.py
    ├── gradual_unfreezing/gradual_unfreezing.py
    ├── limited_data/limited_data.py
    ├── imbalanced/
    │   ├── imbalanced_finetune.py
    │   └── analysis/confusion.py    # post-hoc confusion-matrix diagnostics
    └── lora/
        ├── lora_finetune.py         # shared library: build_lora_resnet50 / _linear_probe / _full_finetune
        ├── train_with_Lora.py       # trains lora / linear_probe / full_finetune
        ├── explore-rank.py          # LoRA rank sweep r ∈ {8,16,32}
        ├── lora_alpha_search.py     # α/r sweep  (⚠ notebook fragment, see caveats)
        ├── lora_lr_search.py        # learning-rate sweep + cosine annealing
        ├── full_tuned_vs_lora.py    # LoRA vs full fine-tuning across data fractions
        ├── plot_results_lora.py     # compute/memory comparison plots
        └── results_{lora,linear_probe,full_finetune}.json   # inputs for plot_results_lora
```

**Design note:** every script writes its plots to `data/figures/<experiment>/` and reads
the dataset from the repository root. Both paths are relative (`../..` for the data root,
`../../data/figures/<experiment>/` for outputs), so **a script only resolves its paths
correctly when run from its own folder** — see [the general rule](#general-rule-run-each-script-from-its-own-folder).

---

## Setup

### 1. Python environment

Python **3.10+** is recommended (developed/tested on 3.13).

```bash
git clone git@github.com:augustfi/Transfer-Learning-LoRa-on-Oxford-IIIT-Pet-Dataset.git
cd Transfer-Learning-LoRa-on-Oxford-IIIT-Pet-Dataset

python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` pulls `torch`, `torchvision`, `numpy`, `scipy`, `scikit-learn`,
`matplotlib`, `Pillow`. (Install the CUDA-matching `torch`/`torchvision` build for your
GPU from <https://pytorch.org> if you want GPU training.)

### 2. The dataset (auto-download)

Every training script calls `torchvision.datasets.OxfordIIITPet(root="../..", download=True)`,
which **downloads the dataset automatically on first run** into `oxford-iiit-pet/` at the
repository root (≈800 MB). You do not need to fetch anything manually.

- The committed `data/images/` and `data/annotations/` are a *reference copy* of the raw
  dataset (torchvision uses its own `oxford-iiit-pet/` layout, so the two live side by side).
- **Exception:** [`limited_data/limited_data.py`](src/limited_data/limited_data.py) uses
  `download=False`. Run any `download=True` script first (e.g. the binary experiment) so the
  dataset is present, then run the limited-data script.

### 3. Hardware

Device is auto-selected in this order: **CUDA → Apple MPS → CPU**. Everything runs on CPU
(slower). The confusion analysis in the imbalanced experiment is CPU-only by design and needs
no GPU.

---

## How the code maps to the report

| Report section | Folder | Script(s) | Backbone | Key figures produced |
|---|---|---|---|---|
| 5.1 Binary classification | `binary_classification/` | `beginning.py` | ResNet-34 | `accuracy.png` |
| 5.2 Breed classification | `breed_classification/` | `multi_class.py` | ResNet-34 | `accuracy.png` |
| 5.2.1 Fine-tuning L layers | `finetune_l_layers/` | `fine_tune_l_layers.py` | ResNet-34 | `compare_all_l_avg.png` |
| 5.2.2 Gradual unfreezing | `gradual_unfreezing/` | `gradual_unfreezing.py` | ResNet-34 | `compare_all_l_run_{0..4}.png` |
| 5.2.3 Limited data & reg. | `limited_data/` | `limited_data.py` | ResNet-34 | `aug_accuracy_limited_data.png`, `aug_loss_limited_data.png` |
| 5.2.4 Imbalanced classes | `imbalanced/` | `imbalanced_finetune.py` + `analysis/confusion.py` | ResNet-34 | `per_class_f1.png`, `confusion_*` heatmaps |
| 5.3 LoRA (rank/α/lr/mem) | `lora/` | `explore-rank.py`, `lora_alpha_search.py`, `lora_lr_search.py`, `train_with_Lora.py`, `plot_results_lora.py` | ResNet-50 | `lora_*_sweep.png`, `comparison_*.png` |
| 5.4 LoRA vs full fine-tune | `lora/` | `full_tuned_vs_lora.py` | ResNet-50 | `full_vs_lora/headline_test_acc_vs_fraction.png` |

---

## Running the experiments

### General rule: run each script from its own folder

Paths are relative, so **always `cd` into the script's directory first**:

```bash
cd src/<experiment_folder>
python <script>.py
```

Running from anywhere else will break dataset loading and/or figure saving. The LoRA scripts
additionally rely on `from lora_finetune import ...`, which only resolves when run from
`src/lora/`.

---

### 5.1 Binary classification — `src/binary_classification/`

Cat-vs-dog with a pre-trained ResNet-34: the final layer is replaced with a 2-way linear head
and fine-tuned (Adam, lr = 1e-3). Only the head is trained.

```bash
cd src/binary_classification
python beginning.py
```

**Output:** `data/figures/binary_classification/accuracy.png` (train/val accuracy).
Reported test accuracy ≈ **0.987** (report Fig. 4). A transient `best_model.pt` checkpoint is
written to the current folder.

---

### 5.2 Breed classification — `src/breed_classification/`

Same recipe as 5.1 but a 37-way head for all breeds (Adam, lr = 1e-3, head-only).

```bash
cd src/breed_classification
python multi_class.py
```

**Output:** `data/figures/breed_classification/accuracy.png`. Reported test accuracy ≈
**0.873** (report Fig. 5). Transient `best_multiclass_model.pt` in the folder.

---

### 5.2.1 Fine-tuning L layers — `src/finetune_l_layers/`

Trains four models that unfreeze the last **L ∈ {1,2,3,4}** ResNet blocks (plus the head) and
compares their test accuracy, averaged over runs.

```bash
cd src/finetune_l_layers
python fine_tune_l_layers.py
```

**Output:** `data/figures/finetune_l_layers/compare_all_l_avg.png` (report Fig. 6). The
archived `compare_all_l.png` in the same folder is a previous run. Finding: **L = 1** (linear
probe) wins. Transient `best_multiclass_model_l{L}.pt` per L.

---

### 5.2.2 Gradual unfreezing — `src/gradual_unfreezing/`

Unfreezes layers progressively during training (`fc` → `layer4` → … → `layer1`, 5 stages ×
5 epochs) with discriminative learning rates, over **5 seeds**.

```bash
cd src/gradual_unfreezing
python gradual_unfreezing.py
```

**Output:** `data/figures/gradual_unfreezing/compare_all_l_run_{0..4}.png` (report Fig. 7),
plus `run_{i}_{train,val,test}.npy` raw histories in the same folder. Reported test accuracy
≈ **0.899 ± 0.005** over 5 seeds.

---

### 5.2.3 Limited data & regularization — `src/limited_data/`

Studies accuracy vs training-data fraction (5% / 10% / 100%) and the effect of augmentation
and L2 weight decay on the linear probe.

```bash
# make sure the dataset is already downloaded (this script uses download=False):
cd src/binary_classification && python beginning.py   # one-time, to fetch the data
cd ../limited_data
python limited_data.py
```

**Output:** `data/figures/limited_data/aug_accuracy_limited_data.png` and
`aug_loss_limited_data.png`. The other figures in that folder
(`base_l2_*`, `l2_*`, `l_layers_*`, `l_effect_*`, `aug_vs_noaug.png`,
`1e-4_training_curves_gradual.png`, report Figs. 8–10) are archived outputs of the original
limited-data notebooks, kept for reference (see [caveats](#known-caveats)).

---

### 5.2.4 Imbalanced classes + confusion analysis — `src/imbalanced/`

The 12 cat breeds are reduced to **20%** of their training data (test set stays balanced), and
four mitigation strategies are compared: `baseline`, `weighted_ce`, `oversampling`,
`weighted_ce+oversampling`. The model matches the report: **ResNet-34 linear probe** (backbone
fully frozen, only the `fc` head trained), **plain Adam (lr = 1e-3)** with an **L1 penalty**
(`L1_LAMBDA = 1e-5`). The whole sweep runs over **5 seeds** (`seed = 42 + i`).

```bash
cd src/imbalanced
python imbalanced_finetune.py
```

This trains all 4 strategies × 5 seeds, then **automatically runs the post-hoc confusion
analysis**. Outputs land in `data/figures/imbalanced/`:

- `compare_imbalance_strategies.png`, `compare_imbalance_loss.png`, `per_class_f1.png`
  (seed-averaged summary figures; report Figs. 11–13)
- `confusion_<strategy>_aggregate_raw.png` and `..._clustered.png` — row-normalized confusion
  matrices, in raw order and reordered so mutually-confused breeds sit in adjacent blocks
- `per_seed/confusion_<strategy>_seed<seed>.png` — one heatmap per (strategy, seed)
- `preds/<strategy>__seed<seed>.npz` + `preds/meta.npz` — raw predictions the analysis reads
- `confusion_findings.txt` — the control pair (Staffordshire ↔ American Pit Bull Terrier) and
  the Maine Coon / Persian / Ragdoll question, with mean ± std and a seed-presence
  (stable-vs-noise) flag; also printed to stdout

**Re-run the analysis alone (no retraining)** once predictions exist:

```bash
cd src/imbalanced
python -m analysis.confusion ../../data/figures/imbalanced/preds ../../data/figures/imbalanced
```

**Verify the analysis logic** (synthetic block-recovery + sanity checks, no data needed):

```bash
cd src/imbalanced
python -m analysis.confusion --selftest
```

---

### 5.3 / 5.4 LoRA experiments — `src/lora/`

All LoRA scripts use **ResNet-50** and share `lora_finetune.py`, so **run them from
`src/lora/`**. `r = 8`, `α = 16`, `conv3` target layers are the report's chosen configuration.

```bash
cd src/lora
```

| Command | What it does | Output (`data/figures/lora/`) |
|---|---|---|
| `python explore-rank.py` | LoRA rank sweep r ∈ {8,16,32}, 3 seeds each | `rank_sweep/*.png`, `rank_sweep/rank_sweep_combined.png` |
| `python lora_lr_search.py` | Peak-LR sweep with cosine annealing | `lora_lr_sweep.png` |
| `python train_with_Lora.py` | Trains a mode (edit the `main(mode=...)` call: `lora` / `linear_probe` / `full_finetune`) | `<mode>_training.png` |
| `python plot_results_lora.py` | Compute/memory/accuracy comparison across the three modes | `comparison_val_acc.png`, `comparison_time.png`, `comparison_memory.png` |
| `python full_tuned_vs_lora.py` | LoRA vs full fine-tuning across data fractions (5/10/50/100%) | `full_vs_lora/headline_test_acc_vs_fraction.png`, `full_vs_lora/combined_train_val_curves.png`, `full_vs_lora/<method>_frac<f>_curves.png` |

`plot_results_lora.py` reads the committed `results_{lora,linear_probe,full_finetune}.json`
(peak memory / per-epoch time / accuracy), so it works without retraining.

> ⚠ `lora_alpha_search.py` (α/r sweep, report Fig. 1 / §5.3.4) is a **notebook fragment** and
> does **not** run standalone — see [caveats](#known-caveats).

---

## Where outputs go

Everything generated lands under `data/figures/<experiment>/`. Nothing is written to the
repository root or into `src/`, except small **transient** best-model checkpoints
(`best_*.pt`) that a few training scripts save next to themselves and reload at the end of the
same run — they are safe to delete.

---

## Known caveats

- **`lora_alpha_search.py` is not standalone.** Its header says it "reuses
  `build_lora_resnet50, train_model, eval_test, dataloaders, device` already defined in earlier
  cells" — it was extracted from a notebook and has no imports for those names. To run it you
  must first define/import those (e.g. reuse `train_with_Lora.py`'s setup). It is kept for
  reference to the report's α/r sweep.
- **Some limited-data figures come from removed notebooks.** The original `.ipynb` files were
  deleted; `limited_data.py` regenerates only `aug_accuracy_limited_data.png` /
  `aug_loss_limited_data.png`. The remaining figures in `data/figures/limited_data/` are
  archived outputs of those notebooks.
- **The imbalanced experiment was aligned to the report.** It previously trained L = 2 blocks
  with AdamW; it now trains a true linear probe with plain Adam + an L1 penalty, matching the
  report's §5.2.4 setup. Re-running produces the report's numbers, not the old ones.
- **Run from the script's own folder** (paths are relative). The LoRA scripts additionally need
  to be run from `src/lora/` for `from lora_finetune import ...` to resolve.

---

## Authors

Group 2 — KTH Royal Institute of Technology (DD2424 Deep Learning in Data Science):

- John Christensen — johnchr@kth.se
- August Filannino — augustfi@kth.se
- Samy Zouggari — zouggari@kth.se
- Lydia Nasser — lhnasser@kth.se
