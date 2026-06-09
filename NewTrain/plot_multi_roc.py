"""
Plot multiple ROC curves into one comparison figure.

Three input modes (freely combinable):
  1) --roc  "Label=path/to/xxx_roc.npz"       pre-computed FPR / TPR
  2) --roc_dir  path/                           auto-discover *_roc.npz
  3) --scores_dir  path/                        auto-discover *_scores.npz

Mode 3 reads genuine / imposter score arrays, computes FPR / TPR on the fly,
and reports the full metric suite (AUC, EER, d-prime, TAR@FAR).
"""

import argparse
import json
import os
import re

import matplotlib
import numpy as np

from metrics_utils import compute_eer, evaluate_from_scores, tar_at_far

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Plot multi-method ROC comparison")
    parser.add_argument("--roc", action="append", default=[],
                        help="Format: Label=path/to/xxx_roc.npz")
    parser.add_argument("--roc_dir", type=str, default=None,
                        help="Auto-load all *_roc.npz under this directory tree")
    parser.add_argument("--scores_dir", type=str, default=None,
                        help="Auto-load all *_scores.npz under this directory tree")
    parser.add_argument("--title", type=str, default="ROC Comparison")
    parser.add_argument("--output", type=str, required=True, help="Output PNG path")
    parser.add_argument("--summary_json", type=str, default=None,
                        help="Optional summary JSON path")
    parser.add_argument("--label_map", type=str, default="label_map.json",
                        help="Label mapping: JSON or inline old=new. Regex keys use re:")
    parser.add_argument("--far_min", type=float, default=1e-4,
                        help="Left x-limit for plot (default: 1e-4)")
    parser.add_argument("--tar_min", type=float, default=0.995,
                        help="Bottom y-limit for plot (default: 0.995)")
    parser.add_argument("--tar_max", type=float, default=1.0,
                        help="Top y-axis tick value (default: 1.0)")
    parser.add_argument("--tar_pad", type=float, default=0.00005,
                        help="Extra headroom above --tar_max (default: 0.002)")
    parser.add_argument("--tar_tick", type=float, default=0.0005,
                        help="Y-axis tick step (0 = auto)")
    parser.add_argument("--mark_every", type=int, default=0,
                        help="Marker interval on curves (0 = automatic)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
#  File discovery
# ---------------------------------------------------------------------------

def parse_roc_item(item):
    if "=" not in item:
        raise ValueError(f"Invalid --roc item: {item}.  Expected Label=path")
    label, path = item.split("=", 1)
    label = label.strip()
    path = path.strip().strip('"').strip("'")
    if not label or not path:
        raise ValueError(f"Invalid --roc item: {item}")
    return label, path


def _discover_files(root_dir, suffix):
    """Walk *root_dir* recursively returning (label, full_path) for every *suffix* file."""
    items = []
    for dirpath, _, filenames in os.walk(root_dir):
        for name in filenames:
            if not name.endswith(suffix):
                continue
            full_path = os.path.join(dirpath, name)
            label = name[:-len(suffix)]
            items.append((label, full_path))
    items.sort(key=lambda x: x[0].lower())
    return items


def load_label_map(arg):
    exact_map = {}
    regex_rules = []

    def add_mapping(src, dst):
        if src.startswith("re:"):
            pattern = src[3:].strip()
            if not pattern:
                raise ValueError("Empty regex pattern in label_map")
            try:
                regex_rules.append((re.compile(pattern), dst))
            except re.error as exc:
                raise ValueError(f"Invalid regex pattern: {pattern}") from exc
        else:
            exact_map[src] = dst

    if not arg:
        return exact_map, regex_rules
    if os.path.isfile(arg):
        with open(arg, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("label_map JSON must be an object: {old: new}")
        for key, value in data.items():
            add_mapping(str(key), str(value))
        return exact_map, regex_rules

    pairs = [p.strip() for p in arg.replace(";", ",").split(",") if p.strip()]
    for item in pairs:
        if "=" in item:
            src, dst = item.split("=", 1)
        elif ":" in item and not item.startswith("re:"):
            src, dst = item.split(":", 1)
        else:
            raise ValueError(f"Invalid --label_map item: {item}. Use old=new")
        src, dst = src.strip(), dst.strip()
        if not src or not dst:
            raise ValueError(f"Invalid --label_map item: {item}")
        add_mapping(src, dst)
    return exact_map, regex_rules


def apply_label_map(label, exact_map, regex_rules):
    if label in exact_map:
        return exact_map[label]
    for regex, value in regex_rules:
        if regex.search(label):
            return value
    return label


# ---------------------------------------------------------------------------
#  Data loading
# ---------------------------------------------------------------------------

def load_curve_from_npz(label, path):
    """Load pre-computed FPR / TPR from a *_roc.npz file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"ROC file not found: {path}")
    data = np.load(path, allow_pickle=True)
    fpr = np.asarray(data["fpr"], dtype=np.float64)
    tpr = np.asarray(data["tpr"], dtype=np.float64)
    auc_value = float(data["auc"]) if "auc" in data else float(np.trapz(tpr, fpr))
    eer = compute_eer(fpr, tpr)
    return {
        "label": label,
        "path": path,
        "fpr": fpr,
        "tpr": tpr,
        "auc": auc_value,
        "eer": eer,
        "d_prime": None,
        "tar_far": {f"TAR_FAR_E{exp}": tar_at_far(fpr, tpr, far)
                    for exp, far in [("6", 1e-6), ("4", 1e-4), ("2", 1e-2)]},
    }


def load_curve_from_scores(label, path):
    """Compute ROC + all metrics from a *_scores.npz file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Scores file not found: {path}")
    data = np.load(path, allow_pickle=True)
    genuine = np.asarray(data["genuine"], dtype=np.float64).ravel()
    imposter = np.asarray(data["imposter"], dtype=np.float64).ravel()
    result = evaluate_from_scores(genuine, imposter)
    return {
        "label": label,
        "path": path,
        "fpr": result["fpr"],
        "tpr": result["tpr"],
        "auc": result["auc"],
        "eer": result["eer"],
        "d_prime": result["d_prime"],
        "tar_far": {k: v for k, v in result.items() if k.startswith("TAR_")},
    }


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    label_map, label_regex = load_label_map(args.label_map)

    entries = []   # list of (label, path, source_kind)

    # --roc
    for item in args.roc:
        label, path = parse_roc_item(item)
        label = apply_label_map(label, label_map, label_regex)
        entries.append((label, path, "roc"))

    # --roc_dir
    if args.roc_dir:
        for label, path in _discover_files(args.roc_dir, "_roc.npz"):
            label = apply_label_map(label, label_map, label_regex)
            entries.append((label, path, "roc"))

    # --scores_dir
    if args.scores_dir:
        for label, path in _discover_files(args.scores_dir, "_scores.npz"):
            label = apply_label_map(label, label_map, label_regex)
            entries.append((label, path, "scores"))

    if not entries:
        raise RuntimeError(
            "No inputs found.  Provide --roc, --roc_dir, or --scores_dir."
        )

    # Deduplicate by label (last wins).
    dedup = {}
    for label, path, kind in entries:
        if label in dedup:
            print(f"Warning: duplicate label after mapping: {label}. Keeping last.")
        dedup[label] = (path, kind)

    # Load curves.
    curves = []
    for label, (path, kind) in dedup.items():
        if kind == "scores":
            c = load_curve_from_scores(label, path)
        else:
            c = load_curve_from_npz(label, path)
        curves.append(c)

    curves.sort(key=lambda x: x["auc"], reverse=True)

    # ----  Plot  ----
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    plt.figure(figsize=(10, 7))
    colors = list(plt.get_cmap("tab10").colors)
    colors += list(plt.get_cmap("Dark2").colors)
    line_styles = ["-", "--", "-.", ":"]
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h", "8", "p", "H", "d", "+"]
    line_width = 2.8
    marker_size = 7
    marker_edge_width = 1.0

    for i, c in enumerate(curves):
        color = colors[i % len(colors)]
        line_style = line_styles[i % len(line_styles)]
        marker = markers[i % len(markers)]
        fpr = np.asarray(c["fpr"], dtype=np.float64)
        tpr = np.asarray(c["tpr"], dtype=np.float64)

        mask = fpr >= args.far_min
        x, y = fpr[mask], tpr[mask]
        mark_every = args.mark_every if args.mark_every > 0 else max(1, x.size // 18)
        kw = dict(
            linewidth=line_width,
            color=color,
            linestyle=line_style,
            marker=marker,
            markevery=mark_every,
            markersize=marker_size,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=marker_edge_width,
            drawstyle="steps-post",
        )

        if x.size == 0:
            x, y = fpr, tpr

        plt.semilogx(x, y, label=c["label"], **kw)

    plt.grid(True, which="both", linestyle="--", alpha=0.35)
    plt.xlim(args.far_min, 1.0)
    y_top = args.tar_max + args.tar_pad if args.tar_pad > 0 else args.tar_max
    plt.ylim(args.tar_min, y_top)
    plt.xlabel("FAR")
    plt.ylabel("TAR")
    #plt.title(args.title)
    if args.tar_tick and args.tar_tick > 0:
        ticks = np.arange(args.tar_min, args.tar_max + args.tar_tick * 0.5, args.tar_tick)
        ticks = ticks[(ticks >= args.tar_min) & (ticks <= args.tar_max)]
        plt.yticks(ticks)
    legend_handlelength = max(2.0, 1.6 + line_width * 0.4)
    legend_markerscale = max(1.0, marker_size / 5.5)
    plt.legend(
        loc="lower right",
        fontsize=8,
        handlelength=legend_handlelength,
        markerscale=legend_markerscale,
        handletextpad=1.0,
        borderpad=0.8,
        labelspacing=1.0,
    )
    plt.tight_layout()
    plt.savefig(args.output, dpi=320)
    plt.close()

    # ----  Text report  ----
    print()
    print("=" * 80)
    print(f"ROC plot saved  ->  {args.output}")
    print("=" * 80)

    # Build header dynamically: include TAR / d-prime columns only when available
    has_dp = any(c["d_prime"] is not None for c in curves)
    has_tar = any(c["tar_far"] for c in curves)
    tar_keys = ["TAR_FAR_E6", "TAR_FAR_E4", "TAR_FAR_E2"] if has_tar else []

    header = f"{'Method':<24s} {'AUC':>8s}  {'EER':>8s}"
    if has_dp:
        header += f"  {'d-prime':>8s}"
    for k in tar_keys:
        header += f"  {k.replace('TAR_FAR_', 'FAR='):>10s}"
    print(header)
    print("-" * len(header))

    for c in curves:
        eer_str = f"{c['eer']:8.4f}" if c["eer"] is not None else f"{'N/A':>8s}"
        line = f"{c['label']:<24s} {c['auc']:8.4f}  {eer_str}"
        if has_dp:
            dp_str = f"{c['d_prime']:8.4f}" if c["d_prime"] is not None else f"{'N/A':>8s}"
            line += f"  {dp_str}"
        for k in tar_keys:
            tv = c["tar_far"].get(k)
            line += f"  {tv:10.4f}" if tv is not None else f"  {'N/A':>10s}"
        print(line)

    # ----  Summary JSON  ----
    if args.summary_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.summary_json)), exist_ok=True)
        summary = {
            "title": args.title,
            "output": args.output,
            "curves": [
                {
                    "label": c["label"],
                    "path": c["path"],
                    "auc": c["auc"],
                    "eer": c["eer"],
                    "d_prime": c["d_prime"],
                    **c["tar_far"],
                }
                for c in curves
            ],
        }
        with open(args.summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\nSummary JSON  ->  {args.summary_json}")

    print("=" * 80)


if __name__ == "__main__":
    main()
