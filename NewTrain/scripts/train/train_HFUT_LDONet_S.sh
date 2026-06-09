#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# LDONet-S 直接训练脚本（HFUT）
#
# 说明：
# 1) 该脚本调用 NewTrain/train_LDONet_S.py。
# 2) 适用于 Windows + Git Bash / WSL / Linux。
# 3) 默认使用 dataset/train_HFUT.txt。
# -----------------------------------------------------------------------------

usage() {
  cat <<'EOF'
用法:
  bash NewTrain/scripts/train/train_HFUT_LDONet_S.sh [选项]

选项:
  --python <path>         Python 可执行文件（默认: python）
  --gpu <id>              GPU ID（默认: 0）
  --epochs <n>            训练轮数（默认: 400）
  --batch-size <n>        batch size（默认: 512）
  --lr <v>                学习率（默认: 0.001）
  --weight-decay <v>      AdamW 权重衰减（默认: 0.0005）
  --warmup-epochs <n>     预热轮数（默认: 10）
  --min-lr <v>            最小学习率（默认: 1e-6）
  --save-interval <n>     保存间隔（默认: 5）
  --train-file <path>     训练索引文件（默认: dataset/train_HFUT.txt）
  --student-path <path>   学生模型恢复训练 checkpoint（可选）
  --help                  显示帮助

示例:
  bash NewTrain/scripts/train/train_HFUT_LDONet_S.sh --gpu 0 --epochs 200
EOF
}

# ---------------------------
# 默认参数
# ---------------------------
PYTHON_EXE="python"
GPU_ID="0"
EPOCH_NUM="400"
BATCH_SIZE="512"
LR="0.001"
WEIGHT_DECAY="0.0005"
WARMUP_EPOCHS="10"
MIN_LR="1e-6"
SAVE_INTERVAL="5"
STUDENT_PATH=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
TRAIN_SCRIPT="${PROJECT_ROOT}/NewTrain/train_LDONet_S.py"
TRAIN_SET_FILE="${PROJECT_ROOT}/dataset/train_HFUT.txt"

DES_PATH="${PROJECT_ROOT}/results/HFUT/LDONet_S/checkpoint/"
PATH_RST="${PROJECT_ROOT}/results/HFUT/LDONet_S/rst_test/"
RUN_ID_NUM="400"

# ---------------------------
# 参数解析
# ---------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      PYTHON_EXE="$2"; shift 2 ;;
    --gpu)
      GPU_ID="$2"; shift 2 ;;
    --epochs)
      EPOCH_NUM="$2"; shift 2 ;;
    --batch-size)
      BATCH_SIZE="$2"; shift 2 ;;
    --lr)
      LR="$2"; shift 2 ;;
    --weight-decay)
      WEIGHT_DECAY="$2"; shift 2 ;;
    --warmup-epochs)
      WARMUP_EPOCHS="$2"; shift 2 ;;
    --min-lr)
      MIN_LR="$2"; shift 2 ;;
    --save-interval)
      SAVE_INTERVAL="$2"; shift 2 ;;
    --train-file)
      TRAIN_SET_FILE="$2"; shift 2 ;;
    --student-path)
      STUDENT_PATH="$2"; shift 2 ;;
    --help|-h)
      usage; exit 0 ;;
    *)
      echo "未知参数: $1" >&2
      usage
      exit 1 ;;
  esac
done

# ---------------------------
# 前置检查
# ---------------------------
if ! command -v "${PYTHON_EXE}" >/dev/null 2>&1; then
  echo "未找到 Python 可执行文件: ${PYTHON_EXE}" >&2
  exit 1
fi

if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
  echo "训练脚本不存在: ${TRAIN_SCRIPT}" >&2
  exit 1
fi

if [[ ! -f "${TRAIN_SET_FILE}" ]]; then
  echo "训练集索引文件不存在: ${TRAIN_SET_FILE}" >&2
  echo "提示：你可以通过 --train-file 指定其他索引文件。" >&2
  exit 1
fi

if [[ -n "${STUDENT_PATH}" ]] && [[ ! -f "${STUDENT_PATH}" ]]; then
  echo "恢复训练 checkpoint 不存在: ${STUDENT_PATH}" >&2
  exit 1
fi

mkdir -p "${DES_PATH}" "${PATH_RST}"

echo "======================================================================"
echo "              LDONet-S 直接训练（HFUT）"
echo "======================================================================"
echo "Project Root   : ${PROJECT_ROOT}"
echo "Python         : ${PYTHON_EXE}"
echo "Train Script   : ${TRAIN_SCRIPT}"
echo "Train Set File : ${TRAIN_SET_FILE}"
echo "GPU ID         : ${GPU_ID}"
echo "Epochs         : ${EPOCH_NUM}"
echo "Batch Size     : ${BATCH_SIZE}"
echo "Learning Rate  : ${LR}"
echo "Weight Decay   : ${WEIGHT_DECAY}"
echo "Warmup Epochs  : ${WARMUP_EPOCHS}"
echo "Min LR         : ${MIN_LR}"
echo "Save Interval  : ${SAVE_INTERVAL}"
echo "Result Path    : ${DES_PATH}"
echo "======================================================================"

# ---------------------------
# 路径转换（WSL + Windows Python.exe 兼容）
# ---------------------------
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

# ---------------------------
# 执行训练
# ---------------------------
CMD=(
  "${PYTHON_EXE}" "${TRAIN_SCRIPT_P}"
  --dataset HFUT
  --id_num "${RUN_ID_NUM}"
  --train_set_file "${TRAIN_SET_FILE_P}"
  --des_path "${DES_PATH_P}"
  --path_rst "${PATH_RST_P}"
  --batch_size "${BATCH_SIZE}"
  --epoch_num "${EPOCH_NUM}"
  --lr "${LR}"
  --weight_decay "${WEIGHT_DECAY}"
  --warmup_epochs "${WARMUP_EPOCHS}"
  --min_lr "${MIN_LR}"
  --save_interval "${SAVE_INTERVAL}"
  --gpu_id "${GPU_ID}"
  --student_weight 0.7
)

if [[ -n "${STUDENT_PATH}" ]]; then
  CMD+=(--student_path "${STUDENT_PATH}")
fi

"${CMD[@]}"

echo ""
echo "训练完成。"
echo "最佳模型通常位于: ${DES_PATH}net_params_best.pth"
