"""
评估所有数据集的 TAR@FAR 指标
"""

import sys
import os
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import metrics
from scipy import interpolate
from scipy.optimize import brentq
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

F_SIZE = 512

def calculate_cosine_similarity(array1, array2):
    """计算余弦相似度"""
    normalized_array1 = array1 / np.linalg.norm(array1, axis=1, keepdims=True)
    normalized_array2 = array2 / np.linalg.norm(array2, axis=1, keepdims=True)
    similarity_matrix = cosine_similarity(normalized_array1, normalized_array2)
    return similarity_matrix

def Inter_classify(feature, name=None, repeat=40):
    """计算类间相似度"""
    label = None
    prob = None

    for i in range(repeat):
        rang1 = np.arange(feature.shape[0])
        rang2 = np.arange(feature.shape[0])
        ridx1 = np.random.randint(low=0, high=feature.shape[1], size=feature.shape[0])
        ridx2 = np.random.randint(low=0, high=feature.shape[1], size=feature.shape[0])

        data1 = feature[rang1, ridx1]
        data2 = feature[rang2, ridx2]

        similarity_matrix = calculate_cosine_similarity(data1, data2)

        idx = np.tril_indices(similarity_matrix.shape[0], -1)
        p = similarity_matrix[idx].tolist()
        l = np.zeros(len(p), dtype=np.int32).tolist()

        pt = similarity_matrix[rang1, rang1].tolist()
        lt = np.ones(len(pt), dtype=np.int32).tolist()

        if label is None:
            label = (l + lt)
            prob = (p + pt)
        else:
            label += (l + lt)
            prob += (p + pt)

    return label, prob

def run_test1(feature_path, dataset_name):
    """计算评估指标"""
    print(f"Processing: {feature_path}")
    
    # 读取特征文件
    with open(feature_path, 'rb') as f:
        data = pickle.load(f)
    
    features = np.array(data, dtype=np.float32)
    
    # 假设特征已经按照类别组织
    # 计算类间相似度
    label, prob = Inter_classify(features, name=dataset_name, repeat=40)
    
    y_test = np.array(label)
    y_pred_prob = np.array(prob)
    
    # 计算ROC
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)
    
    # AUC
    AUC = metrics.auc(fpr, tpr)
    
    # EER
    EER = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
    
    # TAR @ FAR
    TAR_FAR_E1 = brentq(lambda x: 0.1 - interpolate.interp1d(tpr, fpr)(x), 0., 1.)
    TAR_FAR_E2 = brentq(lambda x: 0.01 - interpolate.interp1d(tpr, fpr)(x), 0., 1.)
    TAR_FAR_E3 = brentq(lambda x: 0.001 - interpolate.interp1d(tpr, fpr)(x), 0., 1.)
    TAR_FAR_E4 = brentq(lambda x: 0.0001 - interpolate.interp1d(tpr, fpr)(x), 0., 1.)
    TAR_FAR_E5 = brentq(lambda x: 0.00001 - interpolate.interp1d(tpr, fpr)(x), 0., 1.)
    TAR_FAR_E6 = brentq(lambda x: 0.000001 - interpolate.interp1d(tpr, fpr)(x), 0., 1.)
    
    results = {
        "AUC": float(AUC),
        "EER": float(EER),
        "TAR_FAR_E1": float(TAR_FAR_E1),
        "TAR_FAR_E2": float(TAR_FAR_E2),
        "TAR_FAR_E3": float(TAR_FAR_E3),
        "TAR_FAR_E4": float(TAR_FAR_E4),
        "TAR_FAR_E5": float(TAR_FAR_E5),
        "TAR_FAR_E6": float(TAR_FAR_E6),
    }
    
    print(f"Results for {dataset_name}:")
    print(f"  AUC: {AUC:.4f}")
    print(f"  EER: {EER:.4f}")
    print(f"  TAR@FAR=1e-6: {TAR_FAR_E6:.4f}")
    
    return results

def evaluate_dataset(dataset_name, features_path, output_dir):
    """评估单个数据集"""
    print(f"\n{'='*60}")
    print(f"Evaluating {dataset_name} dataset")
    print(f"{'='*60}")
    
    if not os.path.exists(features_path):
        print(f"Features file not found: {features_path}")
        return None
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 计算指标
    results = run_test1(features_path, dataset_name)
    
    # 保存结果
    output_file = os.path.join(output_dir, f"{dataset_name}_metrics.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {output_file}")
    
    return results

if __name__ == "__main__":
    work_dir = "/home/czk/code/MSFFnet_distillation_Standardization"
    
    datasets = {
        "Blue": {
            "features": f"{work_dir}/results/Blue/SF1Net_KD/results/features.pkl",
            "output": f"{work_dir}/results/Blue/SF1Net_KD/metrics/"
        },
        "HFUT": {
            "features": f"{work_dir}/results/HFUT/SF1Net_KD/results/features.pkl",
            "output": f"{work_dir}/results/HFUT/SF1Net_KD/metrics/"
        },
        "PolyU": {
            "features": f"{work_dir}/results/PolyU/SF1Net_KD/results/features.pkl",
            "output": f"{work_dir}/results/PolyU/SF1Net_KD/metrics/"
        },
        "TJC": {
            "features": f"{work_dir}/results/TJC/SF1Net_KD/results/features.pkl",
            "output": f"{work_dir}/results/TJC/SF1Net_KD/metrics/"
        }
    }
    
    all_results = {}
    
    for dataset_name, paths in datasets.items():
        results = evaluate_dataset(dataset_name, paths["features"], paths["output"])
        if results:
            all_results[dataset_name] = results
    
    # 保存汇总结果
    summary_file = f"{work_dir}/results/all_metrics_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("All evaluations completed!")
    print(f"Summary saved to: {summary_file}")
    print(f"{'='*60}")