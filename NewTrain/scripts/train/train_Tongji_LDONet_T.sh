#!/usr/bin/env bash
set -euo pipefail

# LDONet-T 教师模型训练脚本（Tongji）

usage() {
  cat <<'EOF'
用法:
  bash NewTrain/scripts/train/train_Tongji_LDONet_T.sh [选项]

选项:
  --python <path>      Python 可执行文件（默认: python）
  --gpu <id>           GPU ID（默认: 0）
  --epochs <n>         训练轮数（默认: 400）
  --batch-size <n>     batch size（默认: 256）
  --lr <v>             学习率（默认: 0.0001）
  --weight-decay <v>   权重衰减（默认: 0.05）
  --warmup-epochs <n>  warmup 轮数（默认: 10）
  --min-lr <v>         最小学习率（默认: 1e-6）
  --save-interval <n>  保存间隔（默认: 5）
  --train-file <path>  训练索引文件（默认: dataset/train_Tongji.txt）
  --help               显示帮助
EOF
}

PYTHON_EXE="python"
GPU_ID="0"
EPOCH_NUM="400"
BATCH_SIZE="256"
LR="0.0001"
WEIGHT_DECAY="0.05"
WARMUP_EPOCHS="10"
MIN_LR="1e-6"
SAVE_INTERVAL="5"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
TRAIN_SCRIPT="${PROJECT_ROOT}/NewTrain/train_LDONet_T.py"
TRAIN_SET_FILE="${PROJECT_ROOT}/dataset/train_Tongji.txt"

DES_PATH="${PROJECT_ROOT}/results/Tongji/LDONet_T/checkpoint/"
PATH_RST="${PROJECT_ROOT}/results/Tongji/LDONet_T/rst_test/"
RUN_ID_NUM="300"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)        PYTHON_EXE="$2";    shift 2 ;;
    --gpu)           GPU_ID="$2";        shift 2 ;;
    --epochs)        EPOCH_NUM="$2";     shift 2 ;;
    --batch-size)    BATCH_SIZE="$2";    shift 2 ;;
    --lr)            LR="$2";            shift 2 ;;
    --weight-decay)  WEIGHT_DECAY="$2";  shift 2 ;;
    --warmup-epochs) WARMUP_EPOCHS="$2"; shift 2 ;;
    --min-lr)        MIN_LR="$2";        shift 2 ;;
    --save-interval) SAVE_INTERVAL="$2"; shift 2 ;;
    --train-file)    TRAIN_SET_FILE="$2";shift 2 ;;
    --help|-h)       usage; exit 0 ;;
    *) echo "未知参数: $1" >&2; usage; exit 1 ;;
  esac
done

if ! command -v "${PYTHON_EXE}" >/dev/null 2>&1; then
  echo "未找到 Python: ${PYTHON_EXE}" >&2; exit 1
fi
if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
  echo "训练脚本不存在: ${TRAIN_SCRIPT}" >&2; exit 1
fi
if [[ ! -f "${TRAIN_SET_FILE}" ]]; then
  echo "训练集索引文件不存在: ${TRAIN_SET_FILE}" >&2
  echo "可通过 --train-file 指定其他路径。" >&2; exit 1
fi

mkdir -p "${DES_PATH}" "${PATH_RST}"

# 路径转换（WSL + Windows Python.exe 兼容）
to_win_path() {
  if [[ "${PYTHON_EXE}" == *.exe ]] && command -v wslpath >/dev/null 2>&1; then
    wslpath -w "$1"
  else
    printf '%s' "$1"
  fi
}

TRAIN_SCRIPT_P="$(to_win_path "${TRAIN_SCRIPT}")"
TRAIN_SET_FILE_P="$(to_win_path "${TRAIN_SET_FILE}")"
DES_PATH_P="$(to_win_path "${DES_PATH}")"
PATH_RST_P="$(to_win_path "${PATH_RST}")"

echo "======================================================================"
echo "          LDONet-T 教师模型训练（Tongji）"
echo "======================================================================"
echo "Project Root : ${PROJECT_ROOT}"
echo "Python       : ${PYTHON_EXE}"
echo "Train File   : ${TRAIN_SET_FILE}"
echo "Classes      : ${RUN_ID_NUM}"
echo "GPU          : ${GPU_ID}"
echo "Epochs       : ${EPOCH_NUM}"
echo "Batch Size   : ${BATCH_SIZE}"
echo "LR           : ${LR}"
echo "Warmup       : ${WARMUP_EPOCHS}"
echo "Result Path  : ${DES_PATH}"
echo "======================================================================"

"${PYTHON_EXE}" "${TRAIN_SCRIPT_P}" \
  --id_num "${RUN_ID_NUM}" \
  --train_set_file "${TRAIN_SET_FILE_P}" \
  --des_path "${DES_PATH_P}" \
  --path_rst "${PATH_RST_P}" \
  --batch_size "${BATCH_SIZE}" \
  --epoch_num "${EPOCH_NUM}" \
  --lr "${LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --warmup_epochs "${WARMUP_EPOCHS}" \
  --min_lr "${MIN_LR}" \
  --save_interval "${SAVE_INTERVAL}" \
  --gpu_id "${GPU_ID}"

echo ""
echo "训练完成。最佳模型位于: ${DES_PATH}net_params_best.pth"
