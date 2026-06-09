"""
Visualize the LDONet-T valley-mask mechanism for academic figures.

The script extracts four intermediate maps from a trained LDONet-T model:
large-scale response, small-scale response, valley_mask, and the small-scale
response after valley-mask decoupling. It also saves a difference map and
simple quantitative descriptors so the visualization is not purely subjective.
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms as T


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.dataset import NormSingleROI  # noqa: E402
from models.LDONet_T import LDONet_T  # noqa: E402


DATASETS = ["PolyU", "TJC", "HFUT", "Blue", "Green", "Red"]
TXT_DATASET_STEMS = {
    "PolyU": "PolyUII",
    "TJC": "TJC",
    "HFUT": "HFUT",
    "Blue": "Blue",
    "Green": "Green",
    "Red": "Red",
}
MODEL_DIR_CANDIDATES = {
    "PolyU": ["LDONet_T"],
    "TJC": ["LDONet_T"],
    "HFUT": ["LDONet_T"],
    "Blue": ["LDONet_T"],
    "Green": ["LDONet_T"],
    "Red": ["LDONet_T"],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize large/small responses and valley_mask in LDONet-T."
    )
    parser.add_argument(
        "--train_dataset",
        type=str,
        default="PolyU",
        choices=DATASETS,
        help="Dataset whose open-set LDONet-T checkpoint is used.",
    )
    parser.add_argument(
        "--vis_dataset",
        type=str,
        default=None,
        choices=DATASETS,
        help="Dataset whose images are visualized. Defaults to --train_dataset.",
    )
    parser.add_argument(
        "--protocol",
        type=str,
        default="open",
        choices=["open", "closed"],
        help="Dataset list protocol used when --test_set_file is omitted.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint override. If omitted, resolved from --train_dataset.",
    )
    parser.add_argument(
        "--test_set_file",
        type=str,
        default=None,
        help="Test-list override. If omitted, resolved from --vis_dataset and --protocol.",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Visualize one image directly. If omitted, samples are read from --test_set_file.",
    )
    parser.add_argument(
        "--indices",
        type=str,
        default="10,30,50,70,90",
        help="Comma-separated sample indices in --test_set_file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output override. If omitted, saved under the training dataset result folder.",
    )
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--aggregation",
        type=str,
        default="energy",
        choices=[
            "energy",
            "mean_abs",
            "max_abs",
            "mean",
            "channel",
            "strongest_delta",
            "strongest_small",
            "strongest_large",
        ],
        help="How to reduce CxHxW feature tensors to one heatmap.",
    )
    parser.add_argument(
        "--channel",
        type=int,
        default=0,
        help="Channel index used when --aggregation channel is selected.",
    )
    parser.add_argument(
        "--figure_style",
        type=str,
        default="panel",
        choices=["panel", "overlay", "both"],
        help="panel: separated heatmaps; overlay: heatmaps over ROI; both: save both figures.",
    )
    parser.add_argument(
        "--smooth_kernel",
        type=int,
        default=3,
        help="Odd average-filter kernel for visualization maps. Use 1 to disable smoothing.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="DPI for saved figures.")
    parser.add_argument("--save_npz", action="store_true", help="Save raw maps for each sample.")
    return parser.parse_args()


def resolve_checkpoint(train_dataset):
    candidates = []
    for model_dir in MODEL_DIR_CANDIDATES[train_dataset]:
        base = PROJECT_ROOT / "results" / train_dataset / model_dir
        candidates.extend(
            [
                base / "checkpoint" / "SOTA.pth",
                base / "checkpoint" / "sota.pth",
                base / "SOTA.pth",
                base / "sota.pth",
                base / "checkpoint" / "net_params_best.pth",
            ]
        )

    for path in candidates:
        if path.exists():
            return path

    tried = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        f"Cannot resolve open-set LDONet-T checkpoint for {train_dataset}.\n"
        f"Tried:\n{tried}\n"
        "Pass --checkpoint to use a specific .pth file."
    )


def resolve_test_set(vis_dataset, protocol):
    stem = TXT_DATASET_STEMS[vis_dataset]
    suffix = "_closed" if protocol == "closed" else ""
    candidates = [
        PROJECT_ROOT / "dataset" / f"test_{stem}{suffix}.txt",
        PROJECT_ROOT / f"test_{stem}{suffix}.txt",
    ]
    if vis_dataset == "PolyU" and protocol == "open":
        candidates.append(PROJECT_ROOT / "dataset" / "test_PolyUII_20.txt")

    for path in candidates:
        if path.exists():
            return path

    tried = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        f"Cannot resolve test list for {vis_dataset} ({protocol}).\n"
        f"Tried:\n{tried}\n"
        "Pass --test_set_file to use a specific list file."
    )


def resolve_output_dir(train_dataset, vis_dataset, checkpoint_path):
    model_dir = checkpoint_path.parent.parent.name if checkpoint_path.parent.name == "checkpoint" else checkpoint_path.parent.name
    return (
        PROJECT_ROOT
        / "results"
        / train_dataset
        / model_dir
        / "valley_mask_visualization"
        / f"{vis_dataset}_response"
    )


def extract_state_dict(checkpoint_obj):
    if isinstance(checkpoint_obj, dict):
        for key in ["state_dict", "model_state_dict", "model", "net", "teacher", "student"]:
            value = checkpoint_obj.get(key)
            if isinstance(value, dict):
                return value
        if checkpoint_obj and all(hasattr(v, "shape") for v in checkpoint_obj.values()):
            return checkpoint_obj
    raise ValueError("Unsupported checkpoint format: cannot find a state_dict.")


def strip_module_prefix(state_dict):
    if any(k.startswith("module.") for k in state_dict):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def infer_num_classes(state_dict, override=None):
    if override is not None:
        return override
    arc_weight = state_dict.get("arcface.weight")
    if arc_weight is not None:
        return int(arc_weight.shape[0])
    raise ValueError("Cannot infer num_classes. Please pass --num_classes.")


def load_model(args, device):
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = strip_module_prefix(extract_state_dict(checkpoint))
    num_classes = infer_num_classes(state_dict, args.num_classes)

    model = LDONet_T(label_num=num_classes)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()

    if missing:
        print(f"[WARN] Missing keys: {len(missing)}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {len(unexpected)}")
    print(f"[INFO] Loaded {args.checkpoint}")
    print(f"[INFO] num_classes={num_classes}")
    return model


def read_txt_samples(txt_path, indices):
    samples = []
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    for idx in indices:
        if idx < 0 or idx >= len(lines):
            raise IndexError(f"Index {idx} is out of range for {txt_path} ({len(lines)} lines).")
        parts = lines[idx].split()
        if len(parts) < 2:
            raise ValueError(f"Invalid dataset line {idx}: {lines[idx]}")
        samples.append({"path": parts[0], "label": parts[1], "index": idx})
    return samples


def load_image_tensor(image_path, device, imside=128):
    transform = T.Compose([T.Resize(imside), T.ToTensor(), NormSingleROI(outchannels=1)])
    image = Image.open(image_path).convert("L")
    tensor = transform(image).unsqueeze(0).to(device)
    return tensor


def aggregate_feature(tensor, mode="energy", channel=0):
    tensor = tensor.detach().float().cpu()
    if tensor.ndim == 4:
        tensor = tensor[0]
    if mode.startswith("strongest_"):
        mode = "channel"
    if mode == "energy":
        heatmap = torch.sqrt(torch.mean(tensor.pow(2), dim=0))
    elif mode == "mean_abs":
        heatmap = torch.mean(torch.abs(tensor), dim=0)
    elif mode == "max_abs":
        heatmap = torch.max(torch.abs(tensor), dim=0).values
    elif mode == "mean":
        heatmap = torch.mean(tensor, dim=0)
    elif mode == "channel":
        if channel < 0 or channel >= tensor.shape[0]:
            raise IndexError(f"Channel {channel} is invalid for tensor with {tensor.shape[0]} channels.")
        heatmap = tensor[channel]
    else:
        raise ValueError(f"Unknown aggregation mode: {mode}")
    return heatmap.numpy()


def select_strongest_channel(large, small, small_masked, mode):
    if mode == "strongest_delta":
        score_tensor = torch.abs(small_masked - small)
    elif mode == "strongest_small":
        score_tensor = torch.abs(small)
    elif mode == "strongest_large":
        score_tensor = torch.abs(large)
    else:
        return None

    score = score_tensor.detach().float()
    if score.ndim == 4:
        score = score[0]
    channel_scores = torch.sqrt(torch.mean(score.pow(2), dim=(1, 2)))
    return int(torch.argmax(channel_scores).item())


def smooth_map(array, kernel_size):
    if kernel_size <= 1:
        return array
    if kernel_size % 2 == 0:
        raise ValueError("--smooth_kernel must be odd.")
    tensor = torch.from_numpy(array).float().unsqueeze(0).unsqueeze(0)
    pad = kernel_size // 2
    tensor = F.avg_pool2d(tensor, kernel_size=kernel_size, stride=1, padding=pad)
    return tensor.squeeze(0).squeeze(0).numpy()


def normalize_roi(tensor):
    roi = tensor.detach().float().cpu()[0, 0].numpy()
    valid = roi[np.isfinite(roi)]
    if valid.size == 0:
        return roi
    lo, hi = np.percentile(valid, [1, 99])
    return np.clip((roi - lo) / (hi - lo + 1e-8), 0, 1)


def robust_limits(*arrays, q_low=1, q_high=99):
    values = np.concatenate([a[np.isfinite(a)].reshape(-1) for a in arrays])
    if values.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(values, [q_low, q_high])
    if abs(hi - lo) < 1e-8:
        hi = lo + 1e-8
    return float(lo), float(hi)


def collect_maps(model, image_tensor, args):
    extractor = model.feature_extraction
    with torch.no_grad():
        large = extractor.gabor_large(image_tensor)
        small = extractor.gabor_small(image_tensor)
        valley_mask = extractor._build_suppression_mask(large)
        small_masked = extractor._decouple_small_scale(small, valley_mask)

    selected_channel = select_strongest_channel(large, small, small_masked, args.aggregation)
    channel = args.channel if selected_channel is None else selected_channel
    map_mode = "channel" if selected_channel is not None else args.aggregation

    large_map = aggregate_feature(large, map_mode, channel)
    small_map = aggregate_feature(small, map_mode, channel)
    masked_map = aggregate_feature(small_masked, map_mode, channel)
    diff_map = masked_map - small_map
    delta_abs_map = aggregate_feature(
        torch.abs(small_masked - small),
        "channel" if selected_channel is not None else "mean",
        channel,
    )

    large_map = smooth_map(large_map, args.smooth_kernel)
    small_map = smooth_map(small_map, args.smooth_kernel)
    masked_map = smooth_map(masked_map, args.smooth_kernel)
    diff_map = smooth_map(diff_map, args.smooth_kernel)
    delta_abs_map = smooth_map(delta_abs_map, args.smooth_kernel)

    return {
        "large": large,
        "small": small,
        "valley_mask": valley_mask,
        "small_masked": small_masked,
        "large_map": large_map,
        "small_map": small_map,
        "masked_map": masked_map,
        "diff_map": diff_map,
        "delta_abs_map": delta_abs_map,
        "selected_channel": channel if selected_channel is not None else -1,
    }


def compute_metrics(maps):
    small = maps["small"].detach().float()
    masked = maps["small_masked"].detach().float()
    delta = masked - small

    small_energy = torch.sqrt(torch.mean(small.pow(2))).item()
    masked_energy = torch.sqrt(torch.mean(masked.pow(2))).item()
    delta_abs = torch.mean(torch.abs(delta)).item()
    energy_change = (masked_energy - small_energy) / (small_energy + 1e-8)
    return {
        "small_energy_before": small_energy,
        "small_energy_after": masked_energy,
        "relative_energy_change": energy_change,
        "mean_abs_delta": delta_abs,
    }


def plot_samples(rows, output_path, dpi):
    import matplotlib.pyplot as plt

    col_titles = [
        "ROI",
        "Large-scale response",
        "Small-scale response",
        "Masked small response",
        "Difference",
    ]
    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, len(col_titles), figsize=(15, 2.7 * n_rows), squeeze=False)

    for row_id, row in enumerate(rows):
        maps = row["maps"]
        small_vmin, small_vmax = robust_limits(maps["small_map"], maps["masked_map"])
        large_vmin, large_vmax = robust_limits(maps["large_map"])
        diff_lo, diff_hi = robust_limits(maps["diff_map"], q_low=2, q_high=98)
        diff_abs = max(abs(diff_lo), abs(diff_hi))
        if diff_abs < 1e-8:
            diff_abs = 1e-8

        panels = [
            (row["roi"], "gray", 0, 1),
            (maps["large_map"], "magma", large_vmin, large_vmax),
            (maps["small_map"], "viridis", small_vmin, small_vmax),
            (maps["masked_map"], "viridis", small_vmin, small_vmax),
            (maps["diff_map"], "coolwarm", -diff_abs, diff_abs),
        ]

        for col_id, (array, cmap, vmin, vmax) in enumerate(panels):
            ax = axes[row_id, col_id]
            im = ax.imshow(array, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if row_id == 0:
                ax.set_title(col_titles[col_id], fontsize=10)
            if col_id == 0:
                selected = maps.get("selected_channel", -1)
                channel_text = f"\nch={selected}" if selected >= 0 else ""
                ax.set_ylabel(f"idx={row['index']}\ny={row['label']}{channel_text}", fontsize=9)
            if col_id > 0:
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
                cbar.ax.tick_params(labelsize=6)

    fig.suptitle("LDONet-T Valley-Guided Small-Scale Decoupling", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def normalize_map(array, q_low=2, q_high=98):
    lo, hi = robust_limits(array, q_low=q_low, q_high=q_high)
    return np.clip((array - lo) / (hi - lo + 1e-8), 0, 1)


def plot_overlay_samples(rows, output_path, dpi):
    import matplotlib.pyplot as plt

    col_titles = [
        "ROI",
        "Large response",
        "Small response",
        "Decoupling strength",
        "Masked small",
        "Signed change",
    ]
    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, len(col_titles), figsize=(15, 2.7 * n_rows), squeeze=False)

    for row_id, row in enumerate(rows):
        roi = row["roi"]
        maps = row["maps"]
        large = normalize_map(maps["large_map"])
        small = normalize_map(maps["small_map"])
        masked = normalize_map(maps["masked_map"])
        strength = normalize_map(maps["delta_abs_map"])

        diff_lo, diff_hi = robust_limits(maps["diff_map"], q_low=2, q_high=98)
        diff_abs = max(abs(diff_lo), abs(diff_hi), 1e-8)
        panels = [
            ("gray", None, roi, 0, 1, 1.0),
            ("gray", "magma", large, 0, 1, 0.58),
            ("gray", "viridis", small, 0, 1, 0.58),
            ("gray", "inferno", strength, 0, 1, 0.62),
            ("gray", "viridis", masked, 0, 1, 0.58),
            ("gray", "coolwarm", maps["diff_map"], -diff_abs, diff_abs, 0.60),
        ]

        for col_id, (_, cmap, array, vmin, vmax, alpha) in enumerate(panels):
            ax = axes[row_id, col_id]
            ax.imshow(roi, cmap="gray", vmin=0, vmax=1)
            if cmap is not None:
                im = ax.imshow(array, cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha)
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
                cbar.ax.tick_params(labelsize=6)
            ax.set_xticks([])
            ax.set_yticks([])
            if row_id == 0:
                ax.set_title(col_titles[col_id], fontsize=10)
            if col_id == 0:
                selected = maps.get("selected_channel", -1)
                channel_text = f"\nch={selected}" if selected >= 0 else ""
                ax.set_ylabel(f"idx={row['index']}\ny={row['label']}{channel_text}", fontsize=9)

    fig.suptitle("LDONet-T Valley-Guided Decoupling Overlay", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_single_panel(array, output_path, cmap, vmin, vmax, dpi, colorbar=False):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(3, 3))
    im = ax.imshow(array, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_axis_off()
    if colorbar:
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cbar.ax.tick_params(labelsize=6)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def save_individual_panels(rows, output_dir, dpi):
    panels_dir = output_dir / "individual_panels"
    panels_dir.mkdir(parents=True, exist_ok=True)

    for row in rows:
        maps = row["maps"]
        stem = f"sample_{row['index']}_label_{row['label']}"
        sample_dir = panels_dir / stem
        sample_dir.mkdir(parents=True, exist_ok=True)

        small_vmin, small_vmax = robust_limits(maps["small_map"], maps["masked_map"])
        large_vmin, large_vmax = robust_limits(maps["large_map"])
        diff_lo, diff_hi = robust_limits(maps["diff_map"], q_low=2, q_high=98)
        diff_abs = max(abs(diff_lo), abs(diff_hi), 1e-8)

        panel_specs = [
            ("01_roi", row["roi"], "gray", 0, 1),
            ("02_large_scale_response", maps["large_map"], "magma", large_vmin, large_vmax),
            ("03_small_scale_response", maps["small_map"], "viridis", small_vmin, small_vmax),
            ("04_masked_small_response", maps["masked_map"], "viridis", small_vmin, small_vmax),
            ("05_difference", maps["diff_map"], "coolwarm", -diff_abs, diff_abs),
        ]

        for name, array, cmap, vmin, vmax in panel_specs:
            save_single_panel(
                array,
                sample_dir / f"{name}.png",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                dpi=dpi,
            )

    return panels_dir


def save_metrics(rows, output_dir):
    metric_path = output_dir / "valley_mask_metrics.csv"
    fieldnames = ["index", "label", "path", "selected_channel"] + list(rows[0]["metrics"].keys())
    with open(metric_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            record = {
                "index": row["index"],
                "label": row["label"],
                "path": row["path"],
                "selected_channel": row["maps"].get("selected_channel", -1),
            }
            record.update(row["metrics"])
            writer.writerow(record)
    return metric_path


def save_raw_npz(row, output_dir):
    stem = f"sample_{row['index']}_label_{row['label']}"
    np.savez_compressed(
        output_dir / f"{stem}_maps.npz",
        roi=row["roi"],
        large_map=row["maps"]["large_map"],
        small_map=row["maps"]["small_map"],
        small_masked_map=row["maps"]["masked_map"],
        diff_map=row["maps"]["diff_map"],
        decoupling_strength=row["maps"]["delta_abs_map"],
        selected_channel=row["maps"].get("selected_channel", -1),
    )


def main():
    args = parse_args()
    vis_dataset = args.vis_dataset or args.train_dataset
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else resolve_checkpoint(args.train_dataset)
    test_set_file = Path(args.test_set_file) if args.test_set_file else resolve_test_set(vis_dataset, args.protocol)
    args.checkpoint = str(checkpoint_path)
    args.test_set_file = str(test_set_file)

    output_dir = Path(args.output_dir) if args.output_dir else resolve_output_dir(
        args.train_dataset,
        vis_dataset,
        checkpoint_path,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model = load_model(args, device)

    if args.image:
        samples = [{"path": args.image, "label": "NA", "index": 0}]
    else:
        indices = [int(x.strip()) for x in args.indices.split(",") if x.strip()]
        samples = read_txt_samples(str(test_set_file), indices)

    print(f"[INFO] train_dataset={args.train_dataset}")
    print(f"[INFO] vis_dataset={vis_dataset}, protocol={args.protocol}")
    print(f"[INFO] test_set_file={test_set_file}")
    print(f"[INFO] output_dir={output_dir}")
    print(f"[INFO] aggregation={args.aggregation}")

    rows = []
    for sample in samples:
        if not os.path.exists(sample["path"]):
            raise FileNotFoundError(
                f"Image not found: {sample['path']}\n"
                "Pass --image with an existing file or adjust --test_set_file."
            )
        image_tensor = load_image_tensor(sample["path"], device)
        maps = collect_maps(model, image_tensor, args)
        row = {
            "index": sample["index"],
            "label": sample["label"],
            "path": sample["path"],
            "roi": normalize_roi(image_tensor),
            "maps": maps,
            "metrics": compute_metrics(maps),
        }
        rows.append(row)
        if args.save_npz:
            save_raw_npz(row, output_dir)

    figure_base = output_dir / "ldonet_t_valley_mask_comparison"
    if args.figure_style in ["panel", "both"]:
        plot_samples(rows, figure_base, args.dpi)
        print(f"[OK] Figure saved: {figure_base.with_suffix('.png')}")
        print(f"[OK] Figure saved: {figure_base.with_suffix('.pdf')}")
    if args.figure_style in ["overlay", "both"]:
        overlay_base = output_dir / "ldonet_t_valley_mask_overlay"
        plot_overlay_samples(rows, overlay_base, args.dpi)
        print(f"[OK] Overlay figure saved: {overlay_base.with_suffix('.png')}")
        print(f"[OK] Overlay figure saved: {overlay_base.with_suffix('.pdf')}")
    panels_dir = save_individual_panels(rows, output_dir, args.dpi)
    print(f"[OK] Individual panels saved: {panels_dir}")
    metric_path = save_metrics(rows, output_dir)

    print(f"[OK] Metrics saved: {metric_path}")


if __name__ == "__main__":
    main()
