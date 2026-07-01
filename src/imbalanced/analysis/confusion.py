"""Post-hoc confusion analysis for the imbalanced fine-tune experiment.

This module is purely additive: it never trains, and it depends only on numpy /
scipy / matplotlib (no torch, no torchvision), so it runs on CPU anywhere and
can be re-executed standalone on the prediction dumps produced by
imbalanced_finetune.py.

It answers two questions the per-class F1 numbers can't:

  1. Are Maine Coon / Persian / Ragdoll mutually confused, or is each confused
     with a *different* neighbour?
  2. Confirm the Staffordshire Bull Terrier <-> American Pit Bull Terrier
     confusion. Both are dogs kept at full data, so scarcity is ruled out and
     any confusion there is pure feature similarity -- the natural control that
     isolates similarity from scarcity.

Because the backbone is a frozen linear probe, misclassifications split into
  - frequency/scarcity errors  (misplaced boundary from too few samples;
    what weighted_ce / oversampling target), and
  - feature-overlap errors     (two breeds map to near-identical features, so
    no linear boundary separates them; not fixable by any reweighting).
The confusion matrix + mutual-confusion clustering is what tells them apart.

Inputs (written by imbalanced_finetune.py):
  <preds_dir>/meta.npz                    class_names, cat_class_indices, ...
  <preds_dir>/<strategy>__seed<seed>.npz  y_true, y_pred  (test set)

Run standalone:
  python -m analysis.confusion <preds_dir> <out_dir>
  python -m analysis.confusion --selftest
"""

import os
import glob
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")                       # headless: no display needed
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform


# Breeds the report singles out. Resolved to indices via the dataset-derived
# class_names at runtime -- never hardcode the index, only the (canonical) name.
CONTROL_PAIR = ("Staffordshire Bull Terrier", "American Pit Bull Terrier")
FOCUS_CATS   = ("Maine Coon", "Persian", "Ragdoll")

CAT_LABEL_COLOR = "navy"                     # mirrors the per_class_f1 cat marking


# ---------------------------------------------------------------------------
# Core matrix ops
# ---------------------------------------------------------------------------
def confusion_counts(y_true, y_pred, num_classes):
    """Raw confusion counts C[i, j] = #(true class i predicted as j)."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    C = np.zeros((num_classes, num_classes), dtype=float)
    np.add.at(C, (y_true, y_pred), 1.0)
    return C


def row_normalize(C):
    """Divide each row by its true-class support -> per-row recall distribution.

    Row i then reads "when the true class was i, where did predictions go".
    C[i, i] is the recall of class i. Empty rows (no support) stay all-zero.
    """
    C = np.asarray(C, dtype=float)
    support = C.sum(axis=1, keepdims=True)
    support_safe = np.where(support == 0, 1.0, support)
    return C / support_safe


def symmetrize(Crn):
    """Symmetric *mutual* confusion S = (C + C.T) / 2 with a zeroed diagonal.

    Confusion is directional but "these two look alike" is symmetric, so we
    average the two directions. The diagonal (self-recall) is zeroed so classes
    cluster by whom they are confused *with*, not by how accurate they are.
    """
    S = 0.5 * (Crn + Crn.T)
    np.fill_diagonal(S, 0.0)
    return S


def cluster_order(Crn, method="average"):
    """Leaf ordering that puts mutually-confused breeds in adjacent blocks.

    Cluster on distance D = 1 - S (S = symmetrized mutual confusion). Returns a
    permutation of range(n) from the dendrogram leaf order.
    """
    n = Crn.shape[0]
    if n < 2:
        return list(range(n))
    S = symmetrize(Crn)
    D = 1.0 - S
    np.fill_diagonal(D, 0.0)                 # required: zero self-distance
    D = 0.5 * (D + D.T)                      # guard against fp asymmetry
    condensed = squareform(D, checks=False)
    Z = linkage(condensed, method=method)
    return dendrogram(Z, no_plot=True)["leaves"]


# ---------------------------------------------------------------------------
# Multi-seed aggregation
# ---------------------------------------------------------------------------
def aggregate_over_seeds(rownorm_list, presence_eps=1e-9):
    """Stack per-seed row-normalized matrices -> mean, std, presence count.

    presence[i, j] = number of seeds in which entry (i, j) exceeded
    presence_eps (i.e. at least one such misclassification occurred). It is the
    stable-vs-noise signal: an entry present in all seeds is a finding, one that
    shows up in a single seed is noise.
    """
    stack = np.array(rownorm_list, dtype=float)       # (n_seeds, n, n)
    mean = stack.mean(axis=0)
    std = stack.std(axis=0)
    presence = (stack > presence_eps).sum(axis=0)
    return mean, std, presence, stack.shape[0]


# ---------------------------------------------------------------------------
# Confusion-partner extractor
# ---------------------------------------------------------------------------
def rank_partners(mean, class_names, target_idx, n=5,
                  std=None, presence=None, n_seeds=None):
    """Top-n classes the target is most confused *with* (off-diagonal, ranked).

    Returns a list of dicts: {idx, name, mean, std, presence, n_seeds}. `std` /
    `presence` / `n_seeds` may be None for a single-matrix (per-seed) call.
    """
    row = np.asarray(mean, dtype=float)[target_idx].copy()
    row[target_idx] = -np.inf                         # drop the diagonal
    order = np.argsort(row)[::-1]
    out = []
    for j in order[:n]:
        if not np.isfinite(row[j]) or row[j] <= 0:
            break
        out.append({
            "idx": int(j),
            "name": class_names[j],
            "mean": float(mean[target_idx, j]),
            "std": None if std is None else float(std[target_idx, j]),
            "presence": None if presence is None else int(presence[target_idx, j]),
            "n_seeds": n_seeds,
        })
    return out


def _fmt_partner(p):
    s = f"{p['name']} {p['mean']:.3f}"
    if p["std"] is not None:
        s += f"+/-{p['std']:.3f}"
    if p["presence"] is not None and p["n_seeds"] is not None:
        s += f" [{p['presence']}/{p['n_seeds']} seeds]"
    return s


# ---------------------------------------------------------------------------
# Plotting -- matches the report style (matplotlib, navy cat marking).
# ---------------------------------------------------------------------------
def plot_confusion(mat, class_names, cat_indices, out_path,
                   order=None, title=""):
    """Row-normalized confusion heatmap. `order` reorders rows+cols (clustered
    vs raw). Cat (reduced) breeds get navy tick labels; dogs stay black."""
    n = mat.shape[0]
    order = list(range(n)) if order is None else list(order)
    cat_set = set(int(c) for c in cat_indices)

    M = mat[np.ix_(order, order)]
    names = [class_names[i] for i in order]

    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(M, vmin=0.0, vmax=1.0, cmap="viridis", aspect="equal")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label="P(predicted | true)  (row-normalized)")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_yticklabels(names, fontsize=7)
    for pos, orig_idx in enumerate(order):
        col = CAT_LABEL_COLOR if orig_idx in cat_set else "black"
        ax.get_xticklabels()[pos].set_color(col)
        ax.get_yticklabels()[pos].set_color(col)

    ax.set_xlabel("predicted breed")
    ax.set_ylabel("true breed")
    ax.set_title(title + f"\n({CAT_LABEL_COLOR} labels = cat breeds, reduced to 20%)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Loading the dumps
# ---------------------------------------------------------------------------
def load_meta(preds_dir):
    meta = np.load(os.path.join(preds_dir, "meta.npz"), allow_pickle=True)
    class_names = [str(c) for c in meta["class_names"]]
    cat_indices = [int(i) for i in meta["cat_class_indices"]]
    return class_names, cat_indices


def load_dumps(preds_dir):
    """{strategy: {seed: (y_true, y_pred)}} from <strategy>__seed<seed>.npz."""
    dumps = {}
    for path in sorted(glob.glob(os.path.join(preds_dir, "*__seed*.npz"))):
        base = os.path.basename(path)[:-len(".npz")]
        strat, _, seed_str = base.rpartition("__seed")
        seed = int(seed_str)
        d = np.load(path)
        dumps.setdefault(strat, {})[seed] = (d["y_true"], d["y_pred"])
    return dumps


# ---------------------------------------------------------------------------
# Findings text (control pair + the three cats), stable-vs-noise aware
# ---------------------------------------------------------------------------
def _name_to_idx(class_names, name):
    try:
        return class_names.index(name)
    except ValueError:
        return None


def build_findings(strat, class_names, mean, std, presence, n_seeds, top_n=5):
    """Human-readable answers to the report's specific questions, for one
    strategy's seed-aggregated confusion matrix."""
    lines = [f"===== strategy: {strat}  (mean +/- std over {n_seeds} seeds) ====="]

    # --- control pair: Staffordshire Bull Terrier <-> American Pit Bull Terrier
    a = _name_to_idx(class_names, CONTROL_PAIR[0])
    b = _name_to_idx(class_names, CONTROL_PAIR[1])
    lines.append("")
    lines.append("[control pair -- both dogs at full data, scarcity ruled out]")
    if a is not None and b is not None:
        lines.append(
            f"  true {CONTROL_PAIR[0]} -> pred {CONTROL_PAIR[1]}: "
            f"{mean[a, b]:.3f}+/-{std[a, b]:.3f} [{presence[a, b]}/{n_seeds} seeds]")
        lines.append(
            f"  true {CONTROL_PAIR[1]} -> pred {CONTROL_PAIR[0]}: "
            f"{mean[b, a]:.3f}+/-{std[b, a]:.3f} [{presence[b, a]}/{n_seeds} seeds]")
    else:
        lines.append("  (control-pair breeds not found in class_names)")

    # --- the three cats: mutual, or each toward a different neighbour?
    lines.append("")
    lines.append("[Maine Coon / Persian / Ragdoll -- mutual, or different neighbours?]")
    focus_idx = {c: _name_to_idx(class_names, c) for c in FOCUS_CATS}
    for cat, ci in focus_idx.items():
        if ci is None:
            lines.append(f"  {cat}: not found in class_names")
            continue
        partners = rank_partners(mean, class_names, ci, n=top_n,
                                 std=std, presence=presence, n_seeds=n_seeds)
        lines.append(f"  {cat}  (recall {mean[ci, ci]:.3f}) -> "
                     + ", ".join(_fmt_partner(p) for p in partners))
    # explicit pairwise block among the three cats
    present = {c: i for c, i in focus_idx.items() if i is not None}
    if len(present) >= 2:
        lines.append("  pairwise among the three (true -> pred):")
        for c1, i1 in present.items():
            for c2, i2 in present.items():
                if c1 == c2:
                    continue
                lines.append(
                    f"    {c1} -> {c2}: {mean[i1, i2]:.3f}+/-{std[i1, i2]:.3f} "
                    f"[{presence[i1, i2]}/{n_seeds} seeds]")

    # --- worst residual off-diagonal confusions overall (stable ones first)
    lines.append("")
    lines.append("[largest stable off-diagonal confusions (present in all seeds)]")
    n = mean.shape[0]
    entries = []
    for i in range(n):
        for j in range(n):
            if i != j and presence[i, j] == n_seeds and mean[i, j] > 0:
                entries.append((mean[i, j], std[i, j], i, j))
    entries.sort(reverse=True)
    for m, s, i, j in entries[:10]:
        tag_i = "(cat)" if class_names[i] in _cat_name_set(class_names) else "(dog)"
        lines.append(f"  true {class_names[i]} {tag_i} -> pred {class_names[j]}: "
                     f"{m:.3f}+/-{s:.3f}")
    return "\n".join(lines)


# cat-name set is only needed for the (cat)/(dog) tag in findings; derived from
# the same 12-breed set the training script uses, matched by dataset name.
_CAT_BREEDS = {
    "Abyssinian", "Bengal", "Birman", "Bombay", "British Shorthair",
    "Egyptian Mau", "Maine Coon", "Persian", "Ragdoll",
    "Russian Blue", "Siamese", "Sphynx",
}


def _cat_name_set(class_names):
    return {n for n in class_names if n in _CAT_BREEDS}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_confusion_analysis(preds_dir, out_dir, top_n=5):
    """Load dumps, produce per-seed + aggregate figures, print/save findings."""
    class_names, cat_indices = load_meta(preds_dir)
    num_classes = len(class_names)
    dumps = load_dumps(preds_dir)
    if not dumps:
        raise FileNotFoundError(f"no *__seed*.npz prediction dumps in {preds_dir}")

    os.makedirs(out_dir, exist_ok=True)
    per_seed_dir = os.path.join(out_dir, "per_seed")
    os.makedirs(per_seed_dir, exist_ok=True)

    findings_all = []
    for strat in sorted(dumps):
        seed_map = dumps[strat]
        rownorms = []
        for seed in sorted(seed_map):
            y_true, y_pred = seed_map[seed]
            C = confusion_counts(y_true, y_pred, num_classes)
            Crn = row_normalize(C)
            rownorms.append(Crn)

            # sanity: rows with support sum to ~1
            support = C.sum(axis=1)
            rs = Crn.sum(axis=1)[support > 0]
            assert np.allclose(rs, 1.0, atol=1e-6), \
                f"{strat} seed{seed}: row sums off ({rs.min():.4f}..{rs.max():.4f})"

            # Deliverable #1: per-strategy, per-seed row-normalized heatmap.
            plot_confusion(
                Crn, class_names, cat_indices,
                os.path.join(per_seed_dir, f"confusion_{strat}_seed{seed}.png"),
                order=None,
                title=f"Row-normalized confusion -- {strat}, seed {seed}",
            )

        # Deliverable #4: aggregate over seeds.
        mean, std, presence, n_seeds = aggregate_over_seeds(rownorms)

        # Deliverable #2: aggregate heatmaps, raw order and clustered order.
        plot_confusion(
            mean, class_names, cat_indices,
            os.path.join(out_dir, f"confusion_{strat}_aggregate_raw.png"),
            order=None,
            title=f"Row-normalized confusion (mean of {n_seeds} seeds) -- "
                  f"{strat}, raw order",
        )
        order = cluster_order(mean)
        plot_confusion(
            mean, class_names, cat_indices,
            os.path.join(out_dir, f"confusion_{strat}_aggregate_clustered.png"),
            order=order,
            title=f"Row-normalized confusion (mean of {n_seeds} seeds) -- "
                  f"{strat}, clustered by mutual confusion",
        )

        # Deliverables #3/#4: partner extraction + control pair, stable-vs-noise.
        findings_all.append(
            build_findings(strat, class_names, mean, std, presence, n_seeds, top_n))

    report = "\n\n".join(findings_all)
    print("\n" + "=" * 74)
    print("CONFUSION ANALYSIS")
    print("=" * 74)
    print(report)

    findings_path = os.path.join(out_dir, "confusion_findings.txt")
    with open(findings_path, "w") as f:
        f.write(report + "\n")
    print(f"\nFigures + findings written to {out_dir}")
    print(f"  findings: {findings_path}")
    return report


# ---------------------------------------------------------------------------
# Self-test -- dry-run the clustering on synthetic block-structured data and
# confirm the known mutually-confused blocks are recovered before trusting it
# on real data. Also checks row-normalization and the top-N extractor.
# ---------------------------------------------------------------------------
def _synthetic_blocks(block_sizes, within=0.30, diag=0.6, seed=0):
    """Build a row-normalized confusion matrix with known mutually-confused
    blocks: within a block, off-diagonal mass `within` is spread; classes in
    different blocks are essentially never confused."""
    rng = np.random.default_rng(seed)
    n = sum(block_sizes)
    block_of = np.concatenate([[b] * s for b, s in enumerate(block_sizes)])
    C = np.zeros((n, n))
    for i in range(n):
        C[i, i] = diag
        mates = [j for j in range(n) if j != i and block_of[j] == block_of[i]]
        if mates:
            share = (within / len(mates)) * (0.8 + 0.4 * rng.random(len(mates)))
            for j, s in zip(mates, share):
                C[i, j] = s
        # tiny cross-block leakage so no exact zeros
        others = [j for j in range(n) if block_of[j] != block_of[i]]
        for j in others:
            C[i, j] = 1e-4
    return row_normalize(C), block_of


def selftest():
    print("[selftest] row-normalization ...")
    y_true = np.array([0, 0, 1, 1, 2, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0, 2])
    C = confusion_counts(y_true, y_pred, 3)
    Crn = row_normalize(C)
    assert np.allclose(Crn.sum(axis=1), 1.0), "row sums must be 1"
    # diagonal == per-class recall
    recall = np.array([1/2, 2/2, 2/3])
    assert np.allclose(np.diag(Crn), recall), "diagonal must equal recall"
    print("  ok: rows sum to 1 and diagonal equals recall")

    print("[selftest] clustering recovers known blocks ...")
    block_sizes = [4, 3, 5]                      # three mutually-confused blocks
    # shuffle so the blocks are NOT already contiguous in index order
    rng = np.random.default_rng(7)
    Crn_blocks, block_of = _synthetic_blocks(block_sizes)
    perm = rng.permutation(len(block_of))
    Crn_shuf = Crn_blocks[np.ix_(perm, perm)]
    block_shuf = block_of[perm]
    order = cluster_order(Crn_shuf)
    ordered_blocks = block_shuf[order]
    # each block must appear as one contiguous run in the leaf order
    runs = np.sum(np.diff(ordered_blocks) != 0) + 1
    assert runs == len(block_sizes), (
        f"expected {len(block_sizes)} contiguous blocks, got {runs}: "
        f"{ordered_blocks.tolist()}")
    print(f"  ok: leaf order = {ordered_blocks.tolist()} "
          f"({len(block_sizes)} contiguous blocks recovered)")

    print("[selftest] top-N partner extractor ...")
    names = [f"c{i}" for i in range(len(block_of))]
    # class 0 is in block 0; its top partners must be its block-0 mates
    partners = rank_partners(Crn_blocks, names, target_idx=0, n=3)
    block0 = {i for i, b in enumerate(block_of) if b == 0 and i != 0}
    got = {p["idx"] for p in partners}
    assert got <= block0, f"partners {got} must be block-0 mates {block0}"
    print(f"  ok: class c0 partners = {[p['name'] for p in partners]}")

    print("[selftest] aggregation mean/std/presence ...")
    mean, std, presence, ns = aggregate_over_seeds([Crn_blocks, Crn_blocks])
    assert np.allclose(mean, Crn_blocks) and np.allclose(std, 0.0)
    assert ns == 2
    print("  ok: identical seeds -> zero std, full presence")

    print("\nALL SELFTESTS PASSED")


def main(argv):
    if argv and argv[0] == "--selftest":
        selftest()
        return
    if len(argv) < 2:
        print("usage: python -m analysis.confusion <preds_dir> <out_dir>")
        print("       python -m analysis.confusion --selftest")
        sys.exit(2)
    run_confusion_analysis(argv[0], argv[1])


if __name__ == "__main__":
    main(sys.argv[1:])
