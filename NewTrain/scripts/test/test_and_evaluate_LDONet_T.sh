#!/usr/bin/env bash
set -euo pipefail

# 测试和评估 LDONet-T / LDONet-S 模型
# 用法: ./test_and_evaluate_LDONet_T.sh [dataset] [gpu_id] [model_type]
# 示例:
# bash NewTrain/scripts/test/test_and_evaluate_LDONet_T.sh


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON="${PYTHON:-python}"

to_python_path() {
    local input_path="$1"
    if command -v cygpath >/dev/null 2>&1; then
        cygpath -w "$input_path"
    elif command -v wslpath >/dev/null 2>&1 && [[ "${PYTHON}" == *.exe ]]; then
        wslpath -w "$input_path"
    else
        printf '%s' "$input_path"
    fi
}

DATASET="${1:-Tongji}"   # Blue, Green, NIR, Red, HFUT, PolyU, Tongji
GPU_ID="${2:-0}"
MODEL_TYPE="${3:-teacher}"  # teacher | student

FEATURES_NAME="features.npy"

case "${DATASET}" in
    "Blue")
        NUM_CLASSES=250
        ;;
    "Green")
        NUM_CLASSES=250
        ;;
    "NIR")
        NUM_CLASSES=250
        ;;
    "Red")
        NUM_CLASSES=250
        ;;
    "HFUT")
        NUM_CLASSES=400
        ;;
    "PolyU")
        NUM_CLASSES=193
        ;;
    "Tongji")
        NUM_CLASSES=300
        ;;
    *)
        echo "错误: 不支持的数据集 '${DATASET}'"
        echo "支持的数据集: Blue, Green, NIR, Red, HFUT, PolyU, Tongji"
        exit 1
        ;;
esac

case "${MODEL_TYPE}" in
    "teacher")
        MODEL_DIR="LDONet_T"
        TEST_SCRIPT="test_LDONet_T.py"
        MODEL_NAME="LDONet-T (Teacher)"
        MODEL_TAG="ldonet_t"
        VIT_FLOOR_NUM=32
        CHECKPOINT="${PROJECT_DIR}/results/${DATASET}/${MODEL_DIR}/checkpoint/sota.pth"
        if [[ ! -f "${CHECKPOINT}" ]];then
            CHECKPOINT="${PROJECT_DIR}/results/${DATASET}/${MODEL_DIR}/checkpoint/net_params_best.pth"
        fi
        ;;
    "student")
        MODEL_DIR="LDONet_S_KD"
        TEST_SCRIPT="test_LDONet_S.py"
        MODEL_NAME="LDONet-S (Student)"
        MODEL_TAG="ldonet_s_kd"
        VIT_FLOOR_NUM=6
        CHECKPOINT="${PROJECT_DIR}/results/${DATASET}/${MODEL_DIR}/checkpoint/epoch_198_net_params.pth"
        if [[ ! -f "${CHECKPOINT}" ]]; then
            CHECKPOINT="${PROJECT_DIR}/results/${DATASET}/${MODEL_DIR}/checkpoint/net_params_best.pth"
        fi
        ;;
    *)
        echo "错误: 不支持的 model_type '${MODEL_TYPE}'"
        echo "支持的 model_type: teacher, student"
        exit 1
        ;;
esac

if [[ -z "${TEST_FILE:-}" ]]; then
    if [[ "${DATASET}" == "PolyU" ]]; then
        TEST_FILE="${PROJECT_DIR}/dataset/test_PolyU.txt"
    else
        TEST_FILE="${PROJECT_DIR}/dataset/test_${DATASET}.txt"
    fi
fi

if [[ ! -f "${TEST_FILE}" ]]; then
    ALT_TEST_FILE="${PROJECT_DIR}/dataset/test_${DATASET}_linux.txt"
    if [[ -f "${ALT_TEST_FILE}" ]]; then
        TEST_FILE="${ALT_TEST_FILE}"
    else
        echo "错误: 测试文件不存在: ${TEST_FILE}"
        echo "       也未找到备用文件: ${ALT_TEST_FILE}"
        echo "       请先补齐数据集文件，或通过 TEST_FILE 环境变量指定正确路径。"
        exit 1
    fi
fi

RESULTS_DIR="${PROJECT_DIR}/results/${DATASET}/${MODEL_DIR}/rst_test"
mkdir -p "${RESULTS_DIR}"

if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "错误: 模型文件不存在: ${CHECKPOINT}"
    exit 1
fi

TEST_FILE_PY="$(to_python_path "${TEST_FILE}")"
CHECKPOINT_PY="$(to_python_path "${CHECKPOINT}")"
RESULTS_DIR_PY="$(to_python_path "${RESULTS_DIR}")"
FEATURES_NPY_PY="${RESULTS_DIR_PY}\\${FEATURES_NAME}"
LABELS_NPY_PY="${RESULTS_DIR_PY}\\labels.npy"
FEATURES_PKL_PY="${RESULTS_DIR_PY}\\features.pkl"
RESULT_BASENAME="${DATASET}_${MODEL_TAG}"
RESULT_TXT="${RESULTS_DIR}/${RESULT_BASENAME}_result.txt"
ROC_PNG="${RESULTS_DIR}/${RESULT_BASENAME}_roc.png"

cd "${PROJECT_DIR}/NewTrain"

echo "========================================================================"
echo "测试和评估模型"
echo "========================================================================"
echo "模型类型: ${MODEL_NAME}"
echo "模型角色: ${MODEL_TYPE}"
echo "数据集: ${DATASET}"
echo "类别数: ${NUM_CLASSES}"
echo "GPU ID: ${GPU_ID}"
echo "模型路径: ${CHECKPOINT}"
echo "测试文件: ${TEST_FILE}"
echo "========================================================================"
echo ""

echo "步骤1: 测试模型并提取特征..."
echo "----------------------------------------"

if ! "${PYTHON}" "${TEST_SCRIPT}" \
    --dataset "${DATASET}" \
    --num_classes "${NUM_CLASSES}" \
    --checkpoint "${CHECKPOINT_PY}" \
    --test_set_file "${TEST_FILE_PY}" \
    --output_dir "${RESULTS_DIR_PY}" \
    --gpu_id "${GPU_ID}"; then
    echo "错误: 模型测试失败"
    exit 1
fi

FEATURES_FILE="${RESULTS_DIR}/${FEATURES_NAME}"
LABELS_FILE="${RESULTS_DIR}/labels.npy"

if [[ ! -f "${FEATURES_FILE}" ]]; then
    echo "错误: 特征文件未生成: ${FEATURES_FILE}"
    exit 1
fi

if [[ ! -f "${LABELS_FILE}" ]]; then
    echo "错误: 标签文件未生成: ${LABELS_FILE}"
    exit 1
fi

echo "特征文件输出目录: ${RESULTS_DIR}"
echo ""

echo "步骤2: 转换特征并评估（AUC/EER/TAR@FAR）..."
echo "----------------------------------------"

if ! "${PYTHON}" convert_features.py \
    "${FEATURES_NPY_PY}" \
    "${LABELS_NPY_PY}" \
    "${FEATURES_PKL_PY}" \
    "${NUM_CLASSES}" \
    "${DATASET}_${MODEL_TAG}"; then
    echo "错误: 评估失败"
    exit 1
fi

if [[ -f "${RESULT_TXT}" ]]; then
    echo "结果文件: ${RESULT_TXT}"
else
    echo "警告: 未找到结果文件 ${RESULT_TXT}"
fi

if [[ -f "${ROC_PNG}" ]]; then
    echo "ROC 图像: ${ROC_PNG}"
else
    echo "警告: 未找到 ROC 图像 ${ROC_PNG}"
fi

echo ""
echo "========================================================================"
echo "测试和评估完成！"
echo "========================================================================"
echo "模型: ${MODEL_NAME}"
echo "数据集: ${DATASET}"
echo "结果保存在: ${RESULTS_DIR}"
echo "========================================================================"

echo "完成！"
