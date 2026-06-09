"""Plot genuine/imposter matching score distributions for LDONet-T and LDONet-S-KD.

Usage:
    python plot_matching_distributions.py PolyU

The script prefers precomputed score files if present. Otherwise it uses
feature caches (features.pkl or features.npy + labels.npy) to compute scores
with cosine similarity.
"""

import argparse
import os
import pickle

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _find_dataset_dir(results_root, dataset_name):
    direct = os.path.join(results_root, dataset_name)
    if os.path.isdir(direct):
        return direct

    if not os.path.isdir(results_root):
        return None

    candidates = [d for d in os.listdir(results_root) if os.path.isdir(os.path.join(results_root, d))]
    for d in candidates:
        if d.lower() == dataset_name.lower():
            return os.path.join(results_root, d)
    return None


def _iter_files(root_dir):
    for cur_root, _, files in os.walk(root_dir):
        for fname in files:
            yield cur_root, fname


def _lower_path(path):
    return os.path.normpath(path).lower()


def _match_keywords(name, require_all=None, require_any=None, exclude=None):
    if require_all and not all(k in name for k in require_all):
        return False
    if require_any and not any(k in name for k in require_any):
        return False
    if exclude and any(k in name for k in exclude):
        return False
    return True


def _select_model_roots(dataset_dir):
    model_dirs = []
    for d in os.listdir(dataset_dir):
        full = os.path.join(dataset_dir, d)
        if os.path.isdir(full):
            model_dirs.append(full)

    teacher = []
    student = []
    for d in model_dirs:
        name = _lower_path(os.path.basename(d))
        if name == "ldonet_t":
            teacher.append(d)
            continue
        if name in ("ldonet_s", "ldonet_s_kd"):
            student.append(d)

    if not teacher and not student:
        # fallback: search all subpaths for likely matches
        for cur_root, _, _ in os.walk(dataset_dir):
            name = _lower_path(os.path.basename(cur_root))
            if name == "ldonet_t":
                teacher.append(cur_root)
            elif name in ("ldonet_s", "ldonet_s_kd"):
                student.append(cur_root)

    roots = sorted(set(teacher + student))
    return roots


def _collect_candidates(dataset_dir, selected_roots):
    candidates = []
    pair_score_dirs = set()
    root_norms = [_lower_path(p) for p in selected_roots]

    for cur_root, fname in _iter_files(dataset_dir):
        if root_norms:
            cur_norm = _lower_path(cur_root)
            if not any(r in cur_norm for r in root_norms):
                continue
        path = os.path.join(cur_root, fname)
        if fname.endswith("_scores.npz"):
            candidates.append({"type": "scores_npz", "path": path})
        elif fname.endswith("features.pkl"):
            candidates.append({"type": "features_pkl", "path": path})
        elif fname == "features.npy":
            labels_path = os.path.join(cur_root, "labels.npy")
            if os.path.exists(labels_path):
                candidates.append({"type": "features_npy", "path": path, "labels": labels_path})
            else:
                candidates.append({"type": "features_npy", "path": path, "labels": None})
        elif fname.endswith(".npz") and fname.startswith("scores_"):
            pair_score_dirs.add(cur_root)
        elif fname.endswith(".npz") and fname.endswith("features.npz"):
            candidates.append({"type": "features_npz", "path": path})

    for d in sorted(pair_score_dirs):
        candidates.append({"type": "pair_scores_dir", "path": d})

    return candidates


def _load_scores_npz(path):
    data = np.load(path)
    if "genuine" not in data or "imposter" not in data:
        raise ValueError(f"Missing genuine/imposter arrays in: {path}")
    return data["genuine"].astype(np.float64), data["imposter"].astype(np.float64)


def _load_pair_scores_dir(path):
    positive = []
    negative = []
    for fname in sorted(os.listdir(path)):
        if not (fname.startswith("scores_") and fname.endswith(".npz")):
            continue
        data = np.load(os.path.join(path, fname))
        scores = np.asarray(data["scores"], dtype=np.float64)
        i = int(data["i"])
        j = int(data["j"])
        if i == j:
            positive.append(scores)
        else:
            negative.append(scores)

    genuine = np.concatenate(positive, axis=0) if positive else np.array([], dtype=np.float64)
    imposter = np.concatenate(negative, axis=0) if negative else np.array([], dtype=np.float64)
    return genuine, imposter


def _load_features_pkl(path):
    with open(path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, list):
        return [np.asarray(x) for x in data]
    if isinstance(data, dict) and "features" in data:
        return [np.asarray(x) for x in data["features"]]

    raise ValueError(f"Unsupported pickle format: {path}")


def _load_features_npy(features_path, labels_path):
    features = np.load(features_path)
    labels = np.load(labels_path)
    if features.shape[0] != labels.shape[0]:
        raise ValueError("Feature/label length mismatch")

    grouped = {}
    for feat, lb in zip(features, labels):
        grouped.setdefault(int(lb), []).append(feat)

    return [np.asarray(grouped[k]) for k in sorted(grouped.keys())]


def _load_features_npz(path):
    data = np.load(path, allow_pickle=True)
    if "features" not in data.files or "labels" not in data.files:
        raise ValueError(f"Missing features/labels in: {path}")
    features = data["features"]
    labels = data["labels"]

    grouped = {}
    for feat, lb in zip(features, labels):
        grouped.setdefault(int(lb), []).append(feat)

    return [np.asarray(grouped[k]) for k in sorted(grouped.keys())]


def _normalize_rows(arr):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D feature array, got shape {arr.shape}")
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    return arr / norms


def _sample_within_class_scores(feats, max_pairs, rng):
    n = feats.shape[0]
    if n < 2:
        return np.array([], dtype=np.float64)

    total_pairs = n * (n - 1) // 2
    if max_pairs is None or total_pairs <= max_pairs:
        sims = feats @ feats.T
        iu = np.triu_indices(n, k=1)
        return sims[iu].astype(np.float64)

    k = int(max_pairs)
    idx_i = rng.integers(0, n, size=k)
    idx_j = rng.integers(0, n - 1, size=k)
    idx_j = idx_j + (idx_j >= idx_i)
    scores = np.sum(feats[idx_i] * feats[idx_j], axis=1)
    return scores.astype(np.float64)


def _sample_between_class_scores(a, b, max_pairs, rng):
    na = a.shape[0]
    nb = b.shape[0]
    if na == 0 or nb == 0:
        return np.array([], dtype=np.float64)

    total_pairs = na * nb
    if max_pairs is None or total_pairs <= max_pairs:
        sims = a @ b.T
        return sims.ravel().astype(np.float64)

    k = int(max_pairs)
    idx_a = rng.integers(0, na, size=k)
    idx_b = rng.integers(0, nb, size=k)
    scores = np.sum(a[idx_a] * b[idx_b], axis=1)
    return scores.astype(np.float64)


def _build_scores_from_features(features_by_class, max_pairs, seed):
    rng = np.random.default_rng(seed)
    normed = []
    for feats in features_by_class:
        feats = np.asarray(feats)
        if feats.size == 0:
            normed.append(np.zeros((0, 0), dtype=np.float32))
        else:
            normed.append(_normalize_rows(feats))

    positive = []
    negative = []

    for i, feats_i in enumerate(normed):
        pos_scores = _sample_within_class_scores(feats_i, max_pairs, rng)
        if pos_scores.size:
            positive.append(pos_scores)

        for j in range(i + 1, len(normed)):
            feats_j = normed[j]
            neg_scores = _sample_between_class_scores(feats_i, feats_j, max_pairs, rng)
            if neg_scores.size:
                negative.append(neg_scores)

    genuine = np.concatenate(positive, axis=0) if positive else np.array([], dtype=np.float64)
    imposter = np.concatenate(negative, axis=0) if negative else np.array([], dtype=np.float64)
    return genuine, imposter


def _maybe_subsample(scores, max_samples, rng):
    if max_samples is None or scores.size <= max_samples:
        return scores
    idx = rng.choice(scores.size, size=max_samples, replace=False)
    return scores[idx]


def _silverman_bandwidth(samples):
    samples = np.asarray(samples, dtype=np.float64)
    samples = samples[np.isfinite(samples)]
    n = samples.size
    if n < 2:
        return 0.1

    std = np.std(samples, ddof=1)
    if not np.isfinite(std) or std <= 1e-12:
        return 0.1

    iqr = np.subtract(*np.percentile(samples, [75, 25]))
    sigma = min(std, iqr / 1.34) if iqr > 0 else std
    bw = 1.06 * sigma * (n ** (-1.0 / 5.0))
    return max(bw, 1e-3)


def _gaussian_kde(samples, grid, bw, chunk_size=20000):
    samples = np.asarray(samples, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)
    if samples.size == 0:
        return np.zeros_like(grid, dtype=np.float64)
    if not np.isfinite(bw) or bw <= 0:
        raise ValueError("KDE bandwidth must be > 0")

    inv_bw = 1.0 / bw
    norm = 1.0 / (np.sqrt(2.0 * np.pi) * bw * samples.size)
    out = np.zeros_like(grid, dtype=np.float64)
    for start in range(0, samples.size, chunk_size):
        chunk = samples[start:start + chunk_size]
        diff = (grid[:, None] - chunk[None, :]) * inv_bw
        out += np.sum(np.exp(-0.5 * diff * diff), axis=1)
    return out * norm


def _trim_curve(x, y, support_idx=None, min_floor=0.02):
    if y.size == 0:
        return x, y

    y_clip = np.maximum(y, 0.0)
    if support_idx is not None and support_idx.size > 0:
        lo = int(support_idx[0])
        hi = int(support_idx[-1])
    else:
        threshold = max(min_floor, 0.0)
        idx = np.where(y_clip >= threshold)[0]
        if idx.size == 0:
            return x, y
        lo = int(idx[0])
        hi = int(idx[-1])
    y_trim = y_clip.copy()
    y_trim[:lo] = np.nan
    y_trim[hi + 1:] = np.nan
    y_trim[lo] = max(y_trim[lo], min_floor)
    y_trim[hi] = max(y_trim[hi], min_floor)
    return x, y_trim


def _force_curve_to_right_boundary(x, y, right_bound=1.0):
    """
    强制曲线的最后一个非空点正好落在右边界上
    不改变曲线形状，只移动最后一个点的x坐标
    """
    # 找到所有非NaN的点
    valid_idx = np.where(~np.isnan(y))[0]
    if valid_idx.size == 0:
        return x, y
    
    # 找到最后一个非NaN点的索引
    last_idx = valid_idx[-1]
    
    # 创建x的副本，避免修改原数组
    x_new = x.copy()
    
    # 将最后一个非NaN点的x坐标强制设为右边界
    x_new[last_idx] = right_bound
    
    return x_new, y

def _plot_distribution(genuine, imposter, title, output_path,
                       bins=80, smooth_sigma=None, tail_floor=0.02,
                       kde_bw=None, kde_max_samples=200000, seed=123):
    if genuine.size == 0 or imposter.size == 0:
        raise ValueError("Empty genuine/imposter arrays")

    # Clip to theoretical range
    genuine = np.clip(genuine, -1.0, 1.0)
    imposter = np.clip(imposter, -1.0, 1.0)
    genuine = genuine[np.isfinite(genuine)]
    imposter = imposter[np.isfinite(imposter)]

    left_bound = -0.5
    right_bound = 1.0

    fig, ax = plt.subplots(figsize=(7, 5))

    grid = np.linspace(left_bound, right_bound, bins)
    rng = np.random.default_rng(seed)
    imposter_kde = _maybe_subsample(imposter, kde_max_samples, rng)
    genuine_kde = _maybe_subsample(genuine, kde_max_samples, rng)

    if kde_bw is None:
        kde_bw = smooth_sigma
    if kde_bw is None:
        pooled = np.concatenate([genuine_kde, imposter_kde], axis=0)
        kde_bw = _silverman_bandwidth(pooled)

    kde_i = _gaussian_kde(imposter_kde, grid, kde_bw)
    kde_g = _gaussian_kde(genuine_kde, grid, kde_bw)

    support_i = np.where(kde_i > 0.0)[0]
    support_g = np.where(kde_g > 0.0)[0]

    centers_i, smooth_i = _trim_curve(
        grid, kde_i, support_idx=support_i, min_floor=tail_floor
    )
    centers_g, smooth_g = _trim_curve(
        grid, kde_g, support_idx=support_g, min_floor=tail_floor
    )

    # No tail padding; clip naturally at bounds
    ax.plot(
        centers_i, smooth_i,
        color="#8b0000", linewidth=1.6,
        label="Imposter",
        clip_on=True
    )

    ax.plot(
        centers_g, smooth_g,
        color="#00008b", linewidth=1.6,
        label="Genuine",
        clip_on=True
    )

    ax.set_xlabel("Matching Score")
    ax.set_ylabel("Probability Density")
    ax.legend()
    ax.grid(True, alpha=0.25)

    # Axis limits and ticks
    ax.set_xlim(left_bound, right_bound)
    ax.set_ylim(bottom=0.0)
    ax.margins(x=0)
    ax.autoscale(enable=False, axis='x')
    ax.set_xticks(np.arange(-0.5, 1.0001, 0.25))

    # Do not show dataset title on top
    # ax.set_title(title)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches=None)
    plt.close(fig)

def _sanitize_name(path, base_dir):
    rel = os.path.relpath(path, base_dir)
    safe = rel.replace("..", "").replace(":", "").replace("\\", "_").replace("/", "_")
    return safe.replace(" ", "_")


def _find_labels_for_features(features_path, dataset_dir):
    direct = os.path.join(os.path.dirname(features_path), "labels.npy")
    if os.path.exists(direct):
        return direct

    try:
        feat_len = np.load(features_path, mmap_mode="r").shape[0]
    except Exception:
        return None

    for cur_root, fname in _iter_files(dataset_dir):
        if fname != "labels.npy":
            continue
        candidate = os.path.join(cur_root, fname)
        try:
            labels = np.load(candidate, mmap_mode="r")
        except Exception:
            continue
        if labels.shape[0] == feat_len:
            return candidate
    return None


def main():
    parser = argparse.ArgumentParser(description="Plot genuine/imposter score distributions")
    parser.add_argument("dataset", help="Dataset name, e.g., PolyU")
    parser.add_argument("--bins", type=int, default=120,
                        help="Number of KDE evaluation points across score range")
    parser.add_argument("--smooth_sigma", type=float, default=None,
                        help="Deprecated; use --kde_bw (overrides auto bandwidth if set)")
    parser.add_argument("--kde_bw", type=float, default=None,
                        help="KDE bandwidth; if omitted, uses Silverman rule of thumb")
    parser.add_argument("--kde_max_samples", type=int, default=200000,
                        help="Max samples per curve for KDE (subsampled if exceeded)")
    parser.add_argument("--tail_floor", type=float, default=0.02,
                        help="Minimum tail height after trimming (percentage)")
    parser.add_argument("--max_pairs", type=int, default=200000,
                        help="Max pairs per class-pair when computing from features (sampled if exceeded)")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_root = os.path.join(project_root, "results")
    dataset_root = _find_dataset_dir(results_root, args.dataset)
    if dataset_root is None:
        raise FileNotFoundError(f"Dataset folder not found under results/: {args.dataset}")

    closed_dir = os.path.join(dataset_root, "closed")
    dataset_dir = closed_dir if os.path.isdir(closed_dir) else dataset_root
    if dataset_dir is None:
        raise FileNotFoundError(f"Dataset folder not found under results/: {args.dataset}")

    out_dir = args.output_dir or os.path.join(dataset_dir, "distributions")
    os.makedirs(out_dir, exist_ok=True)

    selected_roots = _select_model_roots(dataset_dir)
    if not selected_roots:
        print("No LDONet-T/LDONet-S folders found under dataset results.")
        return

    candidates = _collect_candidates(dataset_dir, selected_roots)
    if not candidates:
        print("No cached scores/features found. Run evaluation first to generate scores or features.")
        return

    for item in candidates:
        src_type = item["type"]
        path = item["path"]
        try:
            if src_type == "scores_npz":
                genuine, imposter = _load_scores_npz(path)
                source_label = "scores"
            elif src_type == "pair_scores_dir":
                genuine, imposter = _load_pair_scores_dir(path)
                source_label = "pair_scores"
            elif src_type == "features_pkl":
                features_by_class = _load_features_pkl(path)
                genuine, imposter = _build_scores_from_features(features_by_class, args.max_pairs, args.seed)
                source_label = "features_pkl"
            elif src_type == "features_npy":
                labels_path = item["labels"] or _find_labels_for_features(path, dataset_dir)
                if labels_path is None:
                    raise ValueError("labels.npy not found for features.npy")
                features_by_class = _load_features_npy(path, labels_path)
                genuine, imposter = _build_scores_from_features(features_by_class, args.max_pairs, args.seed)
                source_label = "features_npy"
            elif src_type == "features_npz":
                features_by_class = _load_features_npz(path)
                genuine, imposter = _build_scores_from_features(features_by_class, args.max_pairs, args.seed)
                source_label = "features_npz"
            else:
                continue

            if genuine.size == 0 or imposter.size == 0:
                print(f"Skipping {path}: empty scores")
                continue

            name = _sanitize_name(path, dataset_dir)
            title = f"{args.dataset} - {source_label}"
            out_path = os.path.join(out_dir, f"{name}_dist.png")
            kde_bw = args.kde_bw if args.kde_bw is not None else args.smooth_sigma
            _plot_distribution(
                genuine,
                imposter,
                title,
                out_path,
                bins=args.bins,
                smooth_sigma=args.smooth_sigma,
                tail_floor=args.tail_floor,
                kde_bw=kde_bw,
                kde_max_samples=args.kde_max_samples,
                seed=args.seed,
            )
            print(f"Saved: {out_path} (genuine={genuine.size}, imposter={imposter.size})")
        except Exception as exc:
            print(f"Failed {path}: {exc}")


if __name__ == "__main__":
    main()
