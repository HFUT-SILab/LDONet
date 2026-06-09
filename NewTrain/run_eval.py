"""
从保存的features.npy和labels.npy计算EER、AUC、TAR@FAR等验证指标。
用于本地快速评估，不依赖matplotlib GUI。
"""
import sys
import os
import numpy as np
from sklearn import metrics
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import brentq
from scipy import interpolate

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def compute_verification_metrics(features, labels, num_classes, samples_per_class, repeat=20):
    """
    将features和labels重组为 [num_classes, samples_per_class, feature_dim]，
    计算类内（genuine）和类间（imposter）余弦相似度，然后计算EER/AUC等指标。
    """
    feature_dim = features.shape[1]
    features_3d = features.reshape(num_classes, samples_per_class, feature_dim)

    # 类内相似度 (genuine)
    genuine_scores = []
    idx_lower = np.tril_indices(samples_per_class, -1)
    for c in range(num_classes):
        f = features_3d[c]
        f_norm = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)
        sim = f_norm @ f_norm.T
        genuine_scores.extend(sim[idx_lower].tolist())

    # 类间相似度 (imposter)
    imposter_scores = []
    rng = np.random.RandomState(42)
    for _ in range(repeat):
        idx1 = np.arange(num_classes)
        idx2 = np.arange(num_classes)
        s1 = rng.randint(0, samples_per_class, size=num_classes)
        s2 = rng.randint(0, samples_per_class, size=num_classes)
        d1 = features_3d[idx1, s1]
        d2 = features_3d[idx2, s2]
        d1 = d1 / (np.linalg.norm(d1, axis=1, keepdims=True) + 1e-8)
        d2 = d2 / (np.linalg.norm(d2, axis=1, keepdims=True) + 1e-8)
        sim_mat = d1 @ d2.T
        tril_idx = np.tril_indices(num_classes, -1)
        imposter_scores.extend(sim_mat[tril_idx].tolist())

    genuine_scores = np.array(genuine_scores)
    imposter_scores = np.array(imposter_scores)

    all_scores = np.concatenate([genuine_scores, imposter_scores])
    all_labels = np.concatenate([
        np.ones(len(genuine_scores), dtype=np.int32),
        np.zeros(len(imposter_scores), dtype=np.int32),
    ])

    fpr, tpr, thresholds = metrics.roc_curve(all_labels, all_scores)
    auc = metrics.auc(fpr, tpr)

    # EER
    eer = brentq(lambda x: 1.0 - x - interpolate.interp1d(fpr, tpr)(x), 0.0, 1.0)

    # TAR@FAR
    tpr_interp = interpolate.interp1d(fpr, tpr, bounds_error=False, fill_value=(0.0, 1.0))
    fpr_interp = interpolate.interp1d(tpr, fpr, bounds_error=False, fill_value=(1.0, 0.0))

    tar_results = {}
    for k in [1, 2, 3, 4]:
        far_target = 10 ** (-k)
        tar_val = float(tpr_interp(far_target))
        tar_results[f"TAR@FAR=1e-{k}"] = tar_val

    # d-prime
    mu_gen = np.mean(genuine_scores)
    mu_imp = np.mean(imposter_scores)
    std_gen = np.std(genuine_scores)
    std_imp = np.std(imposter_scores)
    pooled_std = np.sqrt((std_gen**2 + std_imp**2) / 2.0)
    d_prime = abs(mu_gen - mu_imp) / (pooled_std + 1e-8)

    return {
        "AUC": auc,
        "EER": eer,
        "d_prime": d_prime,
        "genuine_mean": float(mu_gen),
        "imposter_mean": float(mu_imp),
        "tar_results": tar_results,
        "fpr": fpr,
        "tpr": tpr,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="计算掌纹验证指标")
    parser.add_argument("--features", type=str, required=True, help="features.npy路径")
    parser.add_argument("--labels", type=str, required=True, help="labels.npy路径")
    parser.add_argument("--num_classes", type=int, default=300)
    parser.add_argument("--samples_per_class", type=int, default=20)
    parser.add_argument("--output", type=str, default=None, help="结果保存目录")
    args = parser.parse_args()

    print("=" * 60)
    print("掌纹验证指标评估")
    print("=" * 60)

    features = np.load(args.features)
    labels = np.load(args.labels)
    print(f"特征维度: {features.shape}")
    print(f"类别数量: {len(np.unique(labels))}")

    results = compute_verification_metrics(
        features, labels, args.num_classes, args.samples_per_class
    )

    print(f"\n{'=' * 60}")
    print("验证评估结果:")
    print(f"{'=' * 60}")
    print(f"  AUC    : {results['AUC']:.6f}")
    print(f"  EER    : {results['EER']*100:.4f}%")
    print(f"  d'     : {results['d_prime']:.4f}")
    print(f"  真实分数均值  : {results['genuine_mean']:.4f}")
    print(f"  冒充分数均值  : {results['imposter_mean']:.4f}")
    print()
    for k, v in results['tar_results'].items():
        print(f"  {k}: {v:.6f}")
    print(f"{'=' * 60}")

    if args.output:
        os.makedirs(args.output, exist_ok=True)
        out_file = os.path.join(args.output, "verification_metrics.txt")
        with open(out_file, 'w', encoding='utf-8') as f:
            f.write(f"AUC: {results['AUC']:.6f}\n")
            f.write(f"EER: {results['EER']*100:.4f}%\n")
            f.write(f"d_prime: {results['d_prime']:.4f}\n")
            for k, v in results['tar_results'].items():
                f.write(f"{k}: {v:.6f}\n")
        print(f"\n结果已保存至: {out_file}")

        roc_file = os.path.join(args.output, "roc_data.npz")
        np.savez(roc_file, fpr=results['fpr'], tpr=results['tpr'])
        print(f"ROC数据已保存至: {roc_file}")
