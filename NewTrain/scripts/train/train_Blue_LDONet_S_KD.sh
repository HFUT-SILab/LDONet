#!/usr/bin/env bash
set -euo pipefail

# LDONet-T -> LDONet-S 知识蒸馏训练（Blue）

PYTHON_EXE="python"
GPU_ID="0"
EPOCH_NUM="400"
BATCH_SIZE="512"
LR="0.0015"
WEIGHT_DECAY="0.0005"
WARMUP_EPOCHS="20"
MIN_LR="1e-6"
SAVE_INTERVAL="2"
TRAIN_FILE=""
TEACHER_PATH=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
TRAIN_SCRIPT="${PROJECT_ROOT}/NewTrain/train_LDONet_S_KD.py"

if [[ -z "${TRAIN_FILE}" ]]; then
    TRAIN_FILE="${PROJECT_ROOT}/dataset/train_Blue.txt"
fi

if [[ -z "${TEACHER_PATH}" ]]; then
    TEACHER_PATH="${PROJECT_ROOT}/results/Blue/LDONet_T/checkpoint/net_params_best.pth"
fi

DES_PATH="${PROJECT_ROOT}/results/Blue/LDONet_S_KD/checkpoint/"
PATH_RST="${PROJECT_ROOT}/results/Blue/LDONet_S_KD/rst_test/"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --python)        PYTHON_EXE="$2"; shift 2 ;;
        --gpu)           GPU_ID="$2"; shift 2 ;;
        --epochs)        EPOCH_NUM="$2"; shift 2 ;;
        --batch-size)    BATCH_SIZE="$2"; shift 2 ;;
        --lr)            LR="$2"; shift 2 ;;
        --weight-decay)  WEIGHT_DECAY="$2"; shift 2 ;;
        --warmup-epochs) WARMUP_EPOCHS="$2"; shift 2 ;;
        --min-lr)        MIN_LR="$2"; shift 2 ;;
        --save-interval) SAVE_INTERVAL="$2"; shift 2 ;;
        --train-file)    TRAIN_FILE="$2"; shift 2 ;;
        --teacher-path)  TEACHER_PATH="$2"; shift 2 ;;
        --help|-h) echo "用法: bash $0 [选项]"; exit 0 ;;
        *) echo "未知参数: $1" >&2; exit 1 ;;
    esac
done

if ! command -v "${PYTHON_EXE}" >/dev/null 2>&1; then
    echo "未找到 Python 可执行文件: ${PYTHON_EXE}" >&2; exit 1
fi
if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
    echo "训练脚本不存在: ${TRAIN_SCRIPT}" >&2; exit 1
fi
if [[ ! -f "${TRAIN_FILE}" ]]; then
    echo "训练集索引文件不存在: ${TRAIN_FILE}" >&2; exit 1
fi
if [[ ! -f "${TEACHER_PATH}" ]]; then
    echo "教师模型 checkpoint 不存在: ${TEACHER_PATH}" >&2
    echo "请先用 train_Blue_LDONet_T.sh 训练教师模型，或通过 --teacher-path 指定路径。" >&2
    exit 1
fi

mkdir -p "${DES_PATH}" "${PATH_RST}"

to_win_path() {
  if [[ "${PYTHON_EXE}" == *.exe ]] && command -v wslpath >/dev/null 2>&1; then
    wslpath -w "$1"
  else
    printf '%s' "$1"
  fi
}

TRAIN_SCRIPT_P="$(to_win_path "${TRAIN_SCRIPT}")"
TRAIN_FILE_P="$(to_win_path "${TRAIN_FILE}")"
TEACHER_PATH_P="$(to_win_path "${TEACHER_PATH}")"
DES_PATH_P="$(to_win_path "${DES_PATH}")"
PATH_RST_P="$(to_win_path "${PATH_RST}")"

echo "======================================================================"
echo "   LDONet-T -> LDONet-S 知识蒸馏训练（Blue）"
echo "======================================================================"
echo "Project Root : ${PROJECT_ROOT}"
echo "Python       : ${PYTHON_EXE}"
echo "Train File   : ${TRAIN_FILE}"
echo "Teacher Path : ${TEACHER_PATH}"
echo "GPU ID       : ${GPU_ID}"
echo "Epochs       : ${EPOCH_NUM}"
echo "Batch Size   : ${BATCH_SIZE}"
echo "LR           : ${LR}"
echo "Save Path    : ${DES_PATH}"
echo "======================================================================"

cd "${PROJECT_ROOT}/NewTrain"
"${PYTHON_EXE}" "${TRAIN_SCRIPT_P}" \
    --dataset Blue \
    --num_classes 250 \
    --train_set_file "${TRAIN_FILE_P}" \
    --teacher_path "${TEACHER_PATH_P}" \
    --teacher_weight 0.7 \
    --student_weight 0.7 \
    --batch_size "${BATCH_SIZE}" \
    --epoch_num "${EPOCH_NUM}" \
    --lr "${LR}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --min_lr "${MIN_LR}" \
    --save_interval "${SAVE_INTERVAL}" \
    --gpu_id "${GPU_ID}" \
    --temperature 2.0 \
    --logit_stand \
    --ce_weight 0.1 \
    --kd_weight 9.0 \
    --warmup_epochs "${WARMUP_EPOCHS}" \
    --des_path "${DES_PATH_P}" \
    --path_rst "${PATH_RST_P}" \
    --run_dir run_kd_LDONet_S_KD_Blue

echo ""
echo "✓ Blue 蒸馏训练完成"
echo "结果保存在: ${DES_PATH}"
echo "======================================================================"
