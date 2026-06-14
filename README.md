# LDONet: Lightweight Deployment-Oriented Network for Palmprint Verification

This repository provides the PyTorch implementation of LDONet, a lightweight network for palmprint verification that balances recognition accuracy with parameter efficiency, computational cost, and inference latency.

The code includes the teacher model **LDONet-T**, the lightweight student model **LDONet-S**, and the distilled model **LDONet-S-KD**. Experiments are conducted on PolyU, TJC, HFUT, and the Blue, Red, and Green bands of the PolyU Multispectral Palmprint Database.

## Method

LDONet consists of local texture extraction and global feature fusion:

- **LDONet-T** uses dual-scale learnable Gabor branches to extract directional palmprint textures and employs EDTM and MGFFM to model multi-scale information.
- **LDONet-S** compresses the teacher into a single-scale architecture and adopts the lightweight LMGFFM, substantially reducing parameters and computation while preserving discriminative capability.
- **LDONet-S-KD** uses LDONet-T as the teacher and further improves the lightweight model through knowledge distillation.

During training, an ArcFace classification head is used to learn discriminative representations. During evaluation, palmprint verification is performed using cosine similarity between normalized features.

## Repository Structure

```text
LDONet/
├── dataset/           # Training and test index files
├── distillation/      # KD, DKD, MLKD, and CTKD
├── models/
│   ├── LDONet_T.py    # Teacher model
│   ├── LDONet_S.py    # Student model
│   └── component/     # EDTM, MGFFM, and related modules
├── NewTrain/
│   ├── train_LDONet_T.py
│   ├── train_LDONet_S.py
│   ├── train_LDONet_S_KD.py
│   ├── test_LDONet_T.py
│   ├── test_LDONet_S.py
│   └── convert_features.py
├── results/           # Checkpoints, features, and evaluation results
└── utils/
```

## Requirements

Python 3.9 or later is recommended. Install the PyTorch version compatible with your local CUDA environment.

```bash
pip install torch torchvision numpy pillow scipy scikit-learn matplotlib einops
```

## Data Preparation

The `dataset/` directory contains the index files used for open-set and closed-set experiments:

| Dataset | Classes | Open-set split | Closed-set split |
| --- | ---: | --- | --- |
| Blue | 250 | `train_Blue_real.txt` / `test_Blue_real.txt` | `train_Blue_real_closed.txt` / `test_Blue_real_closed.txt` |
| Green | 250 | `train_Green_real.txt` / `test_Green_real.txt` | `train_Green_real_closed.txt` / `test_Green_real_closed.txt` |
| Red | 250 | `train_Red_real.txt` / `test_Red_real.txt` | `train_Red_real_closed.txt` / `test_Red_real_closed.txt` |
| HFUT | 400 | `train_HFUT_real.txt` / `test_HFUT_real.txt` | `train_HFUT_real_closed.txt` / `test_HFUT_real_closed.txt` |
| PolyU | 193 | `train_PolyU_real.txt` / `test_PolyU_real.txt` | `train_PolyU_real_closed.txt` / `test_PolyU_real_closed.txt` |
| TJC | 300 | `train_Tongji_real.txt` / `test_Tongji_real.txt` | `train_Tongji_real_closed.txt` / `test_Tongji_real_closed.txt` |

Each line follows this format:

```text
/path/to/palmprint/image.jpg label
```

After downloading and organizing the original datasets, replace the image paths in these index files with valid local paths. All input images are loaded in grayscale and resized to `128 × 128`.

## Training

The following examples use the Blue dataset. For other datasets, replace the index file, number of classes, and output directories accordingly.

### LDONet-T

```bash
python NewTrain/train_LDONet_T.py \
  --train_set_file dataset/train_Blue_real.txt \
  --id_num 250 \
  --des_path results/Blue/LDONet_T/checkpoint/ \
  --path_rst results/Blue/LDONet_T/rst_test/ \
  --gpu_id 0
```

### LDONet-S

```bash
python NewTrain/train_LDONet_S.py \
  --dataset Blue \
  --train_set_file dataset/train_Blue_real.txt \
  --id_num 250 \
  --des_path results/Blue/LDONet_S/checkpoint/ \
  --path_rst results/Blue/LDONet_S/rst_test/ \
  --gpu_id 0
```

### LDONet-S-KD

An LDONet-T checkpoint trained on the corresponding dataset is required before training the student model.

```bash
python NewTrain/train_LDONet_S_KD.py \
  --dataset Blue \
  --num_classes 250 \
  --train_set_file dataset/train_Blue_real.txt \
  --teacher_path results/Blue/LDONet_T/checkpoint/net_params_best.pth \
  --des_path results/Blue/LDONet_S_KD/checkpoint/ \
  --path_rst results/Blue/LDONet_S_KD/rst_test/ \
  --distill_method kd \
  --gpu_id 0
```

The current implementation supports `kd`, `dkd`, `mlkd`, and `ctkd`.

## Evaluation

First, extract features from the test set:

```bash
python NewTrain/test_LDONet_S.py \
  --dataset Blue \
  --num_classes 250 \
  --test_set_file dataset/test_Blue_real.txt \
  --checkpoint results/Blue/LDONet_S_KD/checkpoint/net_params_best.pth \
  --output_dir results/Blue/LDONet_S_KD/rst_test/ \
  --gpu_id 0
```

Then compute AUC, EER, and TAR@FAR and generate the ROC curve:

```bash
python NewTrain/convert_features.py \
  results/Blue/LDONet_S_KD/rst_test/features.npy \
  results/Blue/LDONet_S_KD/rst_test/labels.npy \
  results/Blue/LDONet_S_KD/rst_test/features.pkl \
  250 \
  Blue_LDONet_S_KD
```

Higher AUC and TAR values indicate better performance, while a lower EER is preferred. `E1` through `E6` denote TAR at FAR values from `10^-1` to `10^-6`, respectively.

## Experimental Results

### Closed-set Evaluation

The following table reports the closed-set EER (%) of different methods on six datasets.

| Model | PolyU | TJC | HFUT | Blue | Red | Green |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PalmCode | 0.3352 | 0.1178 | 0.3463 | 0.1333 | 0.1467 | 0.1052 |
| CompCode | 0.1094 | 0.0667 | 0.1649 | 0.0527 | 0.1067 | 0.0445 |
| OrdinalCode | 0.1036 | 0.0185 | 0.0203 | 0.0267 | 0.0267 | 0.0393 |
| FusionCode | 0.2313 | 0.1232 | 0.1723 | 0.1897 | 0.1348 | 0.2037 |
| RLOC | 0.0590 | 0.0251 | 0.0443 | 0.0533 | 0.0267 | 0.2000 |
| BOCV | 0.0345 | 0.0253 | 0.0425 | 0.0400 | 0.0267 | 0.0400 |
| SMCC | 0.0405 | 0.0261 | 0.0361 | 0.0667 | 0.0667 | 0.0667 |
| EBOCV | 0.0881 | 0.0519 | 0.1000 | 0.0006 | 0.0495 | 0.0082 |
| DOC | 0.0691 | 0.0405 | 0.1083 | 0.1337 | 0.0933 | 0.1467 |
| DRCC | 0.0682 | 0.0296 | 0.0715 | 0.0465 | 0.0400 | 0.0400 |
| 2TCC | 0.1670 | 0.1556 | 0.2444 | 0.0889 | 0.1467 | 0.0933 |
| MTCC | 0.0345 | 0.0222 | 0.0122 | 0.0400 | 0.0400 | 0.0400 |
| CompNet | 0.0043 | 0.0012 | 0.0056 | 0.0000 | 0.0000 | 0.0000 |
| CO3Net | 0.0009 | 0.0035 | 0.0111 | 0.0000 | 0.0000 | 0.0000 |
| CCNet | **0.0000** | **0.0000** | 0.0003 | **0.0000** | **0.0000** | **0.0000** |
| SF2Net | **0.0000** | **0.0000** | 0.0015 | **0.0000** | **0.0000** | **0.0000** |
| LDONet-T | **0.0000** | **0.0000** | 0.0004 | **0.0000** | **0.0000** | **0.0000** |
| LDONet-S | 0.0024 | 0.0008 | 0.0038 | **0.0000** | **0.0000** | **0.0000** |
| LDONet-S-KD | **0.0000** | 0.0074 | 0.0032 | **0.0000** | **0.0000** | **0.0000** |

### Open-set Evaluation

#### PolyU

| Model | AUC (%) | EER (%) | TAR@E1 (%) | TAR@E2 (%) | TAR@E3 (%) | TAR@E4 (%) | TAR@E5 (%) | TAR@E6 (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CompNet | 99.9774 | 0.5148 | 99.9509 | 99.6673 | 98.7619 | 96.6485 | 93.6488 | 90.7717 |
| CO3Net | 99.9877 | 0.4909 | 99.9945 | 99.7328 | 98.6965 | 96.5612 | 94.2242 | 91.9416 |
| CCNet | 99.9961 | 0.2032 | 99.9945 | 99.9318 | 99.6182 | 98.7919 | 97.4557 | 95.8113 |
| SF2Net | 99.9976 | 0.2083 | 100.0000 | 99.9482 | 99.6700 | 99.0292 | 97.8975 | 96.2749 |
| LDONet-T | 99.9985 | 0.1582 | 100.0000 | 99.9782 | 99.7627 | 99.1819 | 98.3365 | 97.0984 |
| LDONet-S | 99.9960 | 0.2591 | 99.9945 | 99.9318 | 99.4219 | 98.0365 | 96.1167 | 94.2705 |
| **LDONet-S-KD** | **99.9989** | **0.1469** | **100.0000** | **99.9782** | **99.8118** | **99.3537** | **98.3420** | **97.1830** |

#### TJC

| Model | AUC (%) | EER (%) | TAR@E1 (%) | TAR@E2 (%) | TAR@E3 (%) | TAR@E4 (%) | TAR@E5 (%) | TAR@E6 (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CompNet | 99.9882 | 0.4351 | 99.9895 | 99.7298 | 99.0965 | 97.7005 | 95.4526 | 92.6421 |
| CO3Net | 99.9830 | 0.4193 | 99.9614 | 99.7368 | 99.1351 | 98.0421 | 96.5474 | 94.0035 |
| CCNet | 99.9957 | 0.2455 | 99.9947 | 99.9140 | 99.5877 | 99.1193 | 98.3509 | 96.8333 |
| SF2Net | 99.9969 | 0.1877 | 99.9930 | 99.9456 | 99.7263 | 99.2404 | 98.5246 | 97.5807 |
| LDONet-T | 99.9982 | 0.1583 | 99.9965 | 99.9772 | 99.7772 | 99.4316 | 98.9140 | 98.3070 |
| LDONet-S | 99.9881 | 0.3208 | 99.9614 | 99.8000 | 99.4263 | 98.7000 | 97.7930 | 96.6404 |
| **LDONet-S-KD** | **99.9988** | **0.0952** | **99.9965** | **99.9842** | **99.9070** | **99.6421** | **99.1825** | **98.6000** |

#### HFUT

| Model | AUC (%) | EER (%) | TAR@E1 (%) | TAR@E2 (%) | TAR@E3 (%) | TAR@E4 (%) | TAR@E5 (%) | TAR@E6 (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CompNet | 99.9643 | 0.6079 | 99.9184 | 99.5421 | 98.5421 | 96.6618 | 93.9408 | 90.4171 |
| CO3Net | 99.9278 | 0.9490 | 99.8395 | 99.0737 | 97.6737 | 95.4737 | 91.9184 | 86.8171 |
| CCNet | 99.9839 | 0.4539 | 99.9697 | 99.7316 | 99.1039 | 98.2079 | 96.4171 | 92.6961 |
| SF2Net | 99.9790 | 0.4474 | 99.9500 | 99.7053 | 99.1697 | 98.4132 | 97.0092 | 95.5355 |
| **LDONet-T** | **99.9877** | **0.3580** | **99.9711** | **99.8079** | **99.3553** | **98.4868** | **97.2566** | **95.7895** |
| LDONet-S | 99.9520 | 0.6303 | 99.9000 | 99.5342 | 98.4888 | 97.0895 | 94.8105 | 91.5105 |
| LDONet-S-KD | 99.9809 | 0.3777 | 99.9487 | 99.7816 | 99.2789 | 98.4382 | 97.0868 | 94.2566 |

#### Blue

| Model | AUC (%) | EER (%) | TAR@E1 (%) | TAR@E2 (%) | TAR@E3 (%) | TAR@E4 (%) | TAR@E5 (%) | TAR@E6 (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CompNet | 99.9949 | 0.2612 | 100.0000 | 99.8848 | 99.5273 | 98.9091 | 98.0364 | 96.2788 |
| CO3Net | 99.9971 | 0.2545 | 100.0000 | 99.9455 | 99.5333 | 98.5515 | 97.0242 | 95.7758 |
| CCNet | 99.9989 | 0.1126 | 100.0000 | 99.9697 | 99.8848 | **99.7212** | **99.3879** | **99.0000** |
| SF2Net | 99.9989 | 0.1374 | 100.0000 | 99.9636 | 99.8061 | 99.4606 | 99.0242 | 98.4970 |
| LDONet-T | 99.9987 | 0.1335 | 100.0000 | 99.9697 | 99.8364 | 99.5818 | 99.2303 | 98.9212 |
| LDONet-S | 99.9968 | 0.2242 | 100.0000 | 99.9212 | 99.7152 | 99.3515 | 98.7455 | 97.7939 |
| **LDONet-S-KD** | **99.9995** | **0.0988** | **100.0000** | **99.9879** | **99.8970** | 99.6606 | 99.2424 | 98.9455 |

#### Red

| Model | AUC (%) | EER (%) | TAR@E1 (%) | TAR@E2 (%) | TAR@E3 (%) | TAR@E4 (%) | TAR@E5 (%) | TAR@E6 (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CompNet | 99.9993 | 0.0970 | 100.0000 | 99.9818 | 99.9030 | 99.5939 | 99.0667 | 97.9152 |
| CO3Net | 99.9994 | 0.1040 | 100.0000 | 99.9939 | 99.8909 | 99.6242 | 99.1152 | 98.3758 |
| CCNet | 100.0000 | 0.0307 | 100.0000 | 100.0000 | 99.9939 | 99.9091 | 99.7091 | 99.3879 |
| SF2Net | 100.0000 | 0.0334 | 100.0000 | 100.0000 | **100.0000** | 99.9030 | 99.6667 | 99.2182 |
| LDONet-T | 99.9999 | 0.0346 | 100.0000 | 100.0000 | 99.9818 | 99.9152 | 99.7515 | 99.4848 |
| LDONet-S | 99.9999 | 0.0430 | 100.0000 | 100.0000 | 99.9879 | 99.8182 | 99.5515 | 99.3091 |
| **LDONet-S-KD** | **100.0000** | **0.0246** | **100.0000** | **100.0000** | 99.9939 | **99.9212** | **99.8242** | **99.6909** |

#### Green

| Model | AUC (%) | EER (%) | TAR@E1 (%) | TAR@E2 (%) | TAR@E3 (%) | TAR@E4 (%) | TAR@E5 (%) | TAR@E6 (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CompNet | 99.9942 | 0.3135 | 99.9879 | 99.8970 | 99.4364 | 98.5030 | 97.1152 | 95.4606 |
| CO3Net | 99.9942 | 0.3090 | 99.9939 | 99.8424 | 99.4485 | 98.3636 | 96.6061 | 94.5758 |
| CCNet | 99.9991 | 0.1212 | 100.0000 | 99.9697 | 99.8545 | 99.6242 | 99.1697 | 98.6424 |
| SF2Net | 99.9986 | 0.1458 | 100.0000 | 99.9758 | 99.8242 | 99.4788 | 98.8485 | 98.3212 |
| **LDONet-T** | **99.9996** | 0.0978 | 100.0000 | **100.0000** | 99.9030 | 99.5030 | 98.8485 | 98.0788 |
| LDONet-S | 99.9985 | 0.1708 | 100.0000 | 99.9818 | 99.6727 | 99.1091 | 98.2909 | 97.2061 |
| **LDONet-S-KD** | 99.9995 | **0.0907** | **100.0000** | 99.9818 | **99.9212** | **99.6545** | **99.3455** | **99.1394** |

### Model Complexity

| Model | Parameters | GFLOPs | Model Size (MB) | CPU Latency (ms) |
| --- | ---: | ---: | ---: | ---: |
| SF2Net | 69.938M | 1.7448 | 267.316 | 13.819 |
| LDONet-T | 17.869M | 0.7760 | 68.581 | 8.398 |
| **LDONet-S** | **1.191M** | 0.0962 | **4.962** | 2.158 |
| CO3Net | 79.820M | 2.1461 | 305.325 | 17.133 |
| CCNet | 62.710M | 1.3366 | 240.055 | 14.091 |
| CompNet | 5.093M | **0.0902** | 19.636 | **2.004** |

LDONet-S contains only **1.191M** parameters and has a model size of **4.962 MB**. Compared with LDONet-T, it substantially reduces both parameter count and computational cost. With knowledge distillation, LDONet-S-KD also achieves lower EER and higher TAR in the low-FAR region on multiple open-set benchmarks.

## Acknowledgements

This project is implemented with PyTorch. We thank the authors of the palmprint datasets and open-source methods for providing valuable data and code to the research community.
