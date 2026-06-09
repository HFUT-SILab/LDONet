import json
import os
import pickle
import re
from typing import Dict, List, Optional, Tuple

import matplotlib
import numpy as np
from scipy import interpolate
from scipy.optimize import brentq
from sklearn import metrics
from sklearn.metrics.pairwise import cosine_similarity

matplotlib.use('TkAgg', force=True)
import matplotlib.pyplot as plt

F_SIZE = 512


def get_epoch(file_path):
    match = re.search(r'epoch_(\d+)_net', file_path)

    if match:
        epoch_number = int(match.group(1))
        return epoch_number


def readfile(filename: str, c_class: int) -> Tuple[np.ndarray, List[str]]:
    with open(filename, encoding='utf-8') as file:
        lines = file.readlines()
    length = len(lines)

    features = []
    namelist = []

    for i in range(length):
        lst = lines[i].strip().split(' ')
        namelist.append(lst[0])
        features.append(lst[1:])

    features = np.array(features, dtype=np.float32).reshape((-1, c_class, F_SIZE))
    return features, namelist


def getClass(namelist: list, c_class: int) -> list:
    results = []
    for i in range(0, len(namelist), c_class):
        name = namelist[i]
        results.append(os.path.basename(os.path.dirname(name)))

    return results


def calculate_cosine_similarity(array1, array2):
    """
    计算余弦相似度
    :param array1:
    :param array2:
    :return:
    """
    # Normalize the arrays 转为单位向量
    normalized_array1 = array1 / np.linalg.norm(array1, axis=1, keepdims=True)
    normalized_array2 = array2 / np.linalg.norm(array2, axis=1, keepdims=True)

    # Calculate cosine similarity  计算余弦相似度
    similarity_matrix = cosine_similarity(normalized_array1, normalized_array2)

    return similarity_matrix


def Intra_Similarity(feature: np.ndarray) -> list:
    """
    计算类内相似度
    :param feature:
    :return:
    """
    results = []
    idx = np.tril_indices(feature.shape[1], -1)
    for i in range(feature.shape[0]):
        data = feature[i]  # data.shape == (60, 512)
        similarity_matrix = calculate_cosine_similarity(data, data)
        sim = similarity_matrix[idx].tolist()
        results += sim
    return results


def Inter_Similarity(feature, repeat=40):
    prob = []
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

        prob += p

    return prob


def score_distribution(feature, name):
    genuine_scores = Intra_Similarity(feature)
    imposter_scores = Inter_Similarity(feature, repeat=40)

    plt.clf()
    plt.hist(genuine_scores, bins=np.arange(-1, 1, 0.01), label="genuine", color="green", alpha=0.5, density=True)
    plt.hist(imposter_scores, bins=np.arange(-1, 1, 0.01), label="imposter", color="red", alpha=0.5, density=True)
    plt.xlim(-1, 1)
    plt.legend()
    plt.title(f'{name}')
    plt.savefig(f"{name}_distributions.png", dpi=512)

    genuine_scores = np.array(genuine_scores)
    imposter_scores = np.array(imposter_scores)

    print(f'intra stat {name} -> mean: {np.mean(genuine_scores)};  std: {np.std(genuine_scores)}')
    print(f'inter stat {name} -> mean: {np.mean(imposter_scores)};  std: {np.std(imposter_scores)}')


def calculate_performance_split(y_test, y_pred_prob, name, split_id=1000):
    # y_pred_class = np.ones_like(y_test, dtype=y_test.dtype)
    # y_pred_class = np.ones_like(y_test)
    # # confusion matrix
    # confusion = metrics.confusion_matrix(y_test, y_pred_class)
    # # TP TN FP FN
    # TP = confusion[1, 1]
    # TN = confusion[0, 0]
    # FP = confusion[0, 1]
    # FN = confusion[1, 0]
    #
    # # accuracy
    # ACC = (TP+TN) / float(TP+TN+FN+FP)
    # # ACC = metrics.accuracy_score(y_test, y_pred_class) another way

    # # precision
    # PPV = TP / float(TP+FP)
    # # PPV = metrics.precision_score(y_test, y_pred_class)

    # # TPR, sensitivity, recall
    # TPR = TP / float(TP+FN)
    # # TPR = metrics.recall_score(y_test, y_pred_class)

    # # TNR, specificity
    # TNR = TN / float(TN+FP)

    # # FPR
    # FPR = FP / float(TN+FP)
    # # FPR = 1 - TNR

    # # F1 score
    # F1_score = (2*PPV*TPR) / (PPV+TPR)
    # # F1_score = metrics.f1_score(y_test, y_pred_class)

    # ROC
    # IMPORTANT: first argument is true values, second argument is predicted probabilities
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)
    dir_path = os.path.join(os.getcwd(), "TPR_FPR", name, "CO3Net")
    if not os.path.exists(dir_path):
        # 如果路径不存在，则创建路径
        os.makedirs(dir_path)
    np.save(os.path.join(dir_path, "fpr.npy"), fpr)
    np.save(os.path.join(dir_path, "tpr.npy"), tpr)
    # plt.clf()
    # fig = plt.figure(1)
    # plt.plot(fpr, tpr, 'r')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.0])
    # plt.title(f' ROC curve of {split_id}')
    # plt.xlabel('FPR (False Positive Rate)')
    # plt.ylabel('TPR (True Positive Rate)')
    # plt.grid(True)
    # plt.draw()
    # plt.pause(4)
    # plt.savefig('ROC_' + str(split_id) + '.png')
    # plt.close(fig)

    # AUC
    # IMPORTANT: first argument is true values, second argument is predicted probabilities
    # AUC = metrics.roc_auc_score(y_test, y_pred_prob)
    AUC = metrics.auc(fpr, tpr)
    # # calculate cross-validated AUC
    # from sklearn.cross_validation import cross_val_score
    # mean_socre = cross_val_score(logreg, X, y, cv=10, scoring='roc_auc').mean()
    # print(mean_socre)

    # EER
    EER = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)

    # TAR @ FAR = 0.1 / 0.01 / 0.001, FAR = FPR, TAR = TPR
    TAR_FAR_E1 = brentq(lambda x: 0.1 - interpolate.interp1d(tpr, fpr)(x), 0., 1.)
    TAR_FAR_E2 = brentq(lambda x: 0.01 - interpolate.interp1d(tpr, fpr)(x), 0., 1.)
    TAR_FAR_E3 = brentq(lambda x: 0.001 - interpolate.interp1d(tpr, fpr)(x), 0., 1.)
    TAR_FAR_E4 = brentq(lambda x: 0.0001 - interpolate.interp1d(tpr, fpr)(x), 0., 1.)
    TAR_FAR_E5 = brentq(lambda x: 0.00001 - interpolate.interp1d(tpr, fpr)(x), 0., 1.)
    TAR_FAR_E6 = brentq(lambda x: 0.000001 - interpolate.interp1d(tpr, fpr)(x), 0., 1.)

    # return ACC, AUC, TAR_FAR_E1, TAR_FAR_E2, TAR_FAR_E3, fpr, tpr
    results = {
        "AUC":AUC,
        "EER": EER,
        "TAR_FAR_E1": TAR_FAR_E1,
        "TAR_FAR_E2": TAR_FAR_E2,
        "TAR_FAR_E3": TAR_FAR_E3,
        "TAR_FAR_E4": TAR_FAR_E4,
        "TAR_FAR_E5": TAR_FAR_E5,
        "TAR_FAR_E6": TAR_FAR_E6,
    }
    return results


def Inter_classify(feature, name=None, repeat=40):

    label = None
    # label = label.numpy()
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

    results = calculate_performance_split(label, prob, name)

    print('-' * 20 + f'  {name}  ' + '-' * 20)
    for k, v in results.items():
        print(f"{k} --> {v}")
    return results



def Inter_classify_2(feature, name=None):

    id_num, samples_num, _ = feature.shape

    # calculate postive
    postive = []
    idx = np.tril_indices(samples_num, -1)
    for i in range(id_num):
        feats = feature[i]
        similarity_matrix = calculate_cosine_similarity(feats, feats)
        p = similarity_matrix[idx].tolist()
        postive += p

    postive_label = np.ones(len(postive), dtype=np.int32).tolist()

    # calculate negative
    negative = []
    idx = np.tril_indices(samples_num, 0)
    for i in range(id_num-1):
        for j in range(i+1, id_num):
            feats1 = feature[i, :]
            feats2 = feature[j, :]
            similarity_matrix = calculate_cosine_similarity(feats1, feats2)
            p = similarity_matrix.flatten().tolist()
            negative += p

    negative_label = np.zeros(len(negative), dtype=np.int32).tolist()

    # calculate label and prob
    label = postive_label + negative_label
    print(len(label))
    prob = postive + negative

    results = calculate_performance_split(label, prob, name)

    print('-' * 20 + f'  {name}  ' + '-' * 20)
    for k, v in results.items():
        print(f"{k} --> {v}")
    return results




import torch


def run_test1(feature_file_name, name):

    with open(feature_file_name, 'rb') as feature_file:
        # Features are already organized by class: [num_classes, samples_per_class, feature_dim]
        feature_tensor = pickle.load(feature_file)
        # Convert to numpy array
        feature = np.array(feature_tensor)
    print(f"Feature shape: {feature.shape}")

    # 调用 Inter_classify 函数
    # feature = feature[0:100]
    return Inter_classify_2(feature, name)


def demo():
    x = np.random.random(size=(9, 9))
    idx = np.tril_indices(9, -1)

    y = x[idx]
    print(len(y))
    print(y)

def get_son(target_folder_path,  dataset_name="PolyUM_Blue"):
    for root, dirs, files in os.walk(target_folder_path):
        for file in files:
            if file.endswith('.list'):
                file_path = os.path.join(root, file)
                file_name = os.path.splitext(file)[0]
                print(file_path)
                result_json_path = os.path.join(os.path.dirname(target_folder_path), "far_frr", file_name + ".json")
                if not os.path.exists(result_json_path) and (file_name.isdigit() or file_name == 'best'):
                    result = run_test1(file_path, dataset_name)
                    with open(result_json_path, 'w') as json_file:
                        json.dump(result, json_file)



if __name__ == "__main__":
    all_folder_path = [
        r"/home/czk/code/ya/MSFFnet/results/TJC/transDavit/rst_test/score",
    ]
    for folder_path in all_folder_path:
        print(folder_path)
        get_son(folder_path)

