"""
Convert extracted features to metrics format and run ROC-based evaluation.
"""

import os
import pickle
import sys

import matplotlib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from metrics_utils import evaluate_from_scores

matplotlib.use("Agg")
import matplotlib.pyplot as plt

def convert_features_to_metrics_format(features_file, labels_file, output_file, num_classes):
    """
    Convert features to [num_classes, samples_per_class, feature_dim] format
    
    Args:
        features_file: Path to features.npy
        labels_file: Path to labels.npy
        output_file: Path to output pickle file
        num_classes: Number of classes
    """
    # Load features and labels
    features = np.load(features_file)  # [num_samples, feature_dim]
    labels = np.load(labels_file)  # [num_samples]
    
    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Label range: {labels.min()} - {labels.max()}")
    print(f"Number of unique labels: {len(np.unique(labels))}")

    if features.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Feature/label length mismatch: {features.shape[0]} features vs {labels.shape[0]} labels"
        )

    unique_labels = np.unique(labels)
    if len(unique_labels) != num_classes:
        print(
            "Warning: num_classes does not match labels in test file; "
            f"using {len(unique_labels)} observed classes for ROC grouping."
        )

    # Keep variable sample counts intact. This matches test_traditional_method.py:
    # positives are same-label pairs; negatives are different-label pairs.
    features_reorganized = [
        features[labels == label].astype(np.float32, copy=False)
        for label in unique_labels
    ]
    
    # Save as pickle file
    with open(output_file, 'wb') as f:
        pickle.dump(features_reorganized, f)
    
    sample_counts = [int(x.shape[0]) for x in features_reorganized]
    print(f"Reorganized classes: {len(features_reorganized)}")
    print(f"Samples per class: min={min(sample_counts)}, max={max(sample_counts)}")
    print(f"Saved to: {output_file}")
    
    return features_reorganized


def _build_binary_scores(features_reorganized):
    """
    Build binary labels/scores for verification-style ROC evaluation.
    Positive: same class pair.
    Negative: different class pair.
    """
    positive_scores = []
    for feats in features_reorganized:
        samples_num = feats.shape[0]
        if samples_num < 2:
            continue
        intra_idx = np.tril_indices(samples_num, -1)
        sim = cosine_similarity(feats, feats)
        positive_scores.extend(sim[intra_idx].tolist())

    negative_scores = []
    id_num = len(features_reorganized)
    for i in range(id_num - 1):
        feats1 = features_reorganized[i]
        for j in range(i + 1, id_num):
            feats2 = features_reorganized[j]
            sim = cosine_similarity(feats1, feats2)
            negative_scores.extend(sim.ravel().tolist())

    y_true = np.array(
        [1] * len(positive_scores) + [0] * len(negative_scores),
        dtype=np.int32,
    )
    y_score = np.array(positive_scores + negative_scores, dtype=np.float64)
    return y_true, y_score


def evaluate_and_plot_roc(features_reorganized, dataset_name, output_dir):
    y_true, y_score = _build_binary_scores(features_reorganized)

    genuine = y_score[y_true == 1]
    imposter = y_score[y_true == 0]
    ev = evaluate_from_scores(genuine, imposter)

    fpr = ev["fpr"]
    tpr = ev["tpr"]
    results = {
        "AUC": ev["auc"],
        "EER": ev["eer"],
        "d-prime": ev["d_prime"],
        "TAR_FAR_E6": ev["TAR_FAR_E6"],
        "TAR_FAR_E5": ev["TAR_FAR_E5"],
        "TAR_FAR_E4": ev["TAR_FAR_E4"],
        "TAR_FAR_E3": ev["TAR_FAR_E3"],
        "TAR_FAR_E2": ev["TAR_FAR_E2"],
        "TAR_FAR_E1": ev["TAR_FAR_E1"],
    }

    roc_png = os.path.join(output_dir, f"{dataset_name}_roc.png")
    roc_npz = os.path.join(output_dir, f"{dataset_name}_roc.npz")
    far_min = 1e-6
    mask = fpr >= far_min
    x = fpr[mask]
    y = tpr[mask]
    if x.size == 0:
        x = fpr
        y = tpr
    mark_every = max(1, x.size // 18)

    plt.figure(figsize=(8, 6))
    plt.semilogx(
        x,
        y,
        linewidth=1.8,
        marker="o",
        markevery=mark_every,
        markersize=5,
        markerfacecolor="white",
        markeredgewidth=1.1,
        color="#1f77b4",
    )
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.xlim(far_min, 1.0)
    plt.ylim(0.9990, 1.0)
    plt.xticks([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0])
    y_ticks = np.array([0.9990, 0.9992, 0.9994, 0.9996, 0.9998, 1.0])
    plt.yticks(y_ticks, [f"{v:.4f}" if v < 1.0 else "1.0000" for v in y_ticks])
    plt.xlabel("FAR")
    plt.ylabel("TAR")
    plt.title(f"ROC Curve - {dataset_name}")
    plt.tight_layout()
    plt.savefig(roc_png, dpi=300)
    plt.close()

    np.savez_compressed(
        roc_npz,
        fpr=fpr,
        tpr=tpr,
        auc=np.array(results["AUC"], dtype=np.float64),
        method=np.array(dataset_name),
    )

    # Save genuine / imposter scores for downstream reuse (GI histograms, d-prime, etc.)
    scores_npz = os.path.join(output_dir, f"{dataset_name}_scores.npz")
    np.savez_compressed(
        scores_npz,
        genuine=np.asarray(genuine, dtype=np.float64),
        imposter=np.asarray(imposter, dtype=np.float64),
    )

    result_txt = os.path.join(output_dir, f"{dataset_name}_result.txt")
    with open(result_txt, "w", encoding="utf-8") as f:
        f.write(f"Dataset/Model: {dataset_name}\n")
        f.write(f"AUC: {results['AUC'] * 100:.4f}%\n")
        f.write(f"EER: {results['EER'] * 100:.4f}%\n")
        f.write(f"d-prime: {results['d-prime']}\n")
        f.write(f"TAR@FAR_E6: {results['TAR_FAR_E6'] * 100:.4f}%\n")
        f.write(f"TAR@FAR_E5: {results['TAR_FAR_E5'] * 100:.4f}%\n")
        f.write(f"TAR@FAR_E4: {results['TAR_FAR_E4'] * 100:.4f}%\n")
        f.write(f"TAR@FAR_E3: {results['TAR_FAR_E3'] * 100:.4f}%\n")
        f.write(f"TAR@FAR_E2: {results['TAR_FAR_E2'] * 100:.4f}%\n")
        f.write(f"TAR@FAR_E1: {results['TAR_FAR_E1'] * 100:.4f}%\n")
        f.write(f"ROC Figure: {roc_png}\n")
        f.write(f"ROC Data: {roc_npz}\n")

    print("\n" + "=" * 60)
    print(f"Evaluation completed for {dataset_name}")
    print("=" * 60)
    print(f"AUC: {results['AUC'] * 100:.4f}%")
    print(f"EER: {results['EER'] * 100:.4f}%")
    print(f"d-prime: {results['d-prime']}")
    print(f"TAR@FAR_E6: {results['TAR_FAR_E6'] * 100:.4f}%")
    print(f"TAR@FAR_E5: {results['TAR_FAR_E5'] * 100:.4f}%")
    print(f"TAR@FAR_E4: {results['TAR_FAR_E4'] * 100:.4f}%")
    print(f"TAR@FAR_E3: {results['TAR_FAR_E3'] * 100:.4f}%")
    print(f"TAR@FAR_E2: {results['TAR_FAR_E2'] * 100:.4f}%")
    print(f"TAR@FAR_E1: {results['TAR_FAR_E1'] * 100:.4f}%")
    print(f"ROC curve saved to: {roc_png}")
    print(f"ROC data saved to: {roc_npz}")
    print(f"Result text saved to: {result_txt}")
    print("=" * 60)

    return results, roc_png, result_txt

if __name__ == "__main__":
    if len(sys.argv) < 6:
        print("Usage: python convert_features.py <features_file> <labels_file> <output_file> <num_classes> <dataset_name>")
        print("Example: python convert_features.py features.npy labels.npy features.pkl 250 Blue")
        sys.exit(1)

    features_file = sys.argv[1]
    labels_file = sys.argv[2]
    output_file = sys.argv[3]
    num_classes = int(sys.argv[4])
    dataset_name = sys.argv[5]
    output_dir = os.path.dirname(os.path.abspath(output_file))
    
    # Convert features
    features_reorganized = convert_features_to_metrics_format(features_file, labels_file, output_file, num_classes)
    
    print("\n" + "=" * 60)
    print(f"Running metrics evaluation for {dataset_name}")
    print("=" * 60)

    evaluate_and_plot_roc(features_reorganized, dataset_name, output_dir)
