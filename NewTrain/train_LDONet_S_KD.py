"""
Knowledge Distillation: LDONet-T (Teacher) -> LDONet-S (Student)
Teacher: dual-scale dual-branch LDONet-T
Student: single-scale single-branch LDONet-S
"""

import os
import argparse
import time
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler

from models.dataset import MyDataset
from models.LDONet_T import LDONet_T as TeacherNet
from models.LDONet_S import LDONet_S as StudentNet
from distillation.ctkd import CTKD
from distillation.dkd import DKD
from distillation.kd import KD
from distillation.mlkd import MLKD
from distillation.config import DistillationConfig


DISTILLER_REGISTRY = {
    "kd": KD,
    "dkd": DKD,
    "mlkd": MLKD,
    "ctkd": CTKD,
}


def build_distiller(method, student, teacher, cfg):
    method = method.lower()
    if method not in DISTILLER_REGISTRY:
        available = ", ".join(sorted(DISTILLER_REGISTRY))
        raise ValueError(f"Unknown distillation method: {method}. Available: {available}")
    return DISTILLER_REGISTRY[method](student, teacher, cfg)


def fit(epoch, distiller, data_loader, cfg, optimizer):
    """Train one epoch with knowledge distillation."""
    distiller.train()
    running_loss_ce = 0.0
    running_loss_kd = 0.0
    running_loss_total = 0.0
    running_correct = 0
    total = 0

    for data, target in data_loader:
        data = data.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        batch_size = target.size(0)

        optimizer.zero_grad()

        # Forward pass with KD losses.
        logits_student, losses_dict = distiller(image=data, target=target, epoch=epoch)

        loss_ce = losses_dict['loss_ce']
        loss_kd = losses_dict['loss_kd']

        # KD warmup to avoid unstable early optimization.
        warmup_factor = min(epoch / cfg.WARMUP_EPOCHS, 1.0)
        loss = loss_ce + warmup_factor * loss_kd

        loss.backward()
        optimizer.step()

        preds = logits_student.data.max(dim=1, keepdim=True)[1]
        running_correct += preds.eq(target.data.view_as(preds)).sum().item()

        running_loss_ce += loss_ce.item() * batch_size
        running_loss_kd += loss_kd.item() * batch_size
        running_loss_total += loss.item() * batch_size
        total += batch_size

    loss_ce = running_loss_ce / total
    loss_kd = running_loss_kd / total
    loss_total = running_loss_total / total
    accuracy = (100.0 * running_correct) / total

    if epoch % 10 == 0:
        print(
            f'epoch {epoch}: \tCE loss: {loss_ce:.5f} \tKD loss: {loss_kd:.5f} '
            f'\tTotal loss: {loss_total:.5f} \tAccuracy: {running_correct}/{total} ({accuracy:.3f}%)'
        )

    return loss_ce, loss_kd, loss_total, accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LDONet-T -> LDONet-S Knowledge Distillation"
    )

    # Dataset parameters
    parser.add_argument("--dataset", type=str, default="Blue", choices=["Blue", "Green", "HFUT", "PolyU", "Tongji", "Red"])
    parser.add_argument("--num_classes", type=int, default=250)
    parser.add_argument("--train_set_file", type=str, default="../dataset/train_Blue_linux.txt")

    # Model parameters
    parser.add_argument("--teacher_weight", type=float, default=0.7)
    parser.add_argument("--student_weight", type=float, default=0.7)

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epoch_num", type=int, default=400)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.005)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--save_interval", type=int, default=5)
    parser.add_argument("--gpu_id", type=str, default='0')

    # Distillation parameters
    parser.add_argument("--teacher_path", type=str, required=True,
                        help="Path to LDONet-T teacher checkpoint")
    parser.add_argument("--student_path", type=str, default=None,
                        help="Path to student checkpoint (for resume)")
    parser.add_argument("--distill_method", type=str, default="kd",
                        choices=sorted(DISTILLER_REGISTRY.keys()))
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--logit_stand", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable or disable logit Z-score standardization.")
    parser.add_argument("--ce_weight", type=float, default=0.1)
    parser.add_argument("--kd_weight", type=float, default=9.0)
    parser.add_argument("--dkd_alpha", type=float, default=1.0)
    parser.add_argument("--dkd_beta", type=float, default=8.0)
    parser.add_argument("--mlkd_temperatures", type=float, nargs="*", default=[2.0, 3.0, 4.0])
    parser.add_argument("--mlkd_cc_weight", type=float, default=1.0)
    parser.add_argument("--mlkd_bc_weight", type=float, default=1.0)
    parser.add_argument("--ctkd_min_temp", type=float, default=1.0)
    parser.add_argument("--ctkd_max_temp", type=float, default=10.0)
    parser.add_argument("--warmup_epochs", type=int, default=20)

    # Output paths
    parser.add_argument("--des_path", type=str,
                        default="../results/Blue/LDONet_S_KD/checkpoint/")
    parser.add_argument("--path_rst", type=str,
                        default="../results/Blue/LDONet_S_KD/rst_test/")
    parser.add_argument("--run_dir", type=str, default="run_kd_LDONet_S")

    args = parser.parse_args()

    cfg = DistillationConfig.from_args(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = torch.device("cuda")

    os.makedirs(args.des_path, exist_ok=True)
    os.makedirs(args.path_rst, exist_ok=True)

    print(f"\nLoading dataset: {args.train_set_file}")
    trainset = MyDataset(txt=args.train_set_file, transforms=None, train=True,
                         imside=128, outchannels=1)
    data_loader = DataLoader(dataset=trainset, batch_size=args.batch_size,
                             num_workers=4, shuffle=True, pin_memory=True)
    print(f"Dataset: {len(trainset)} samples")

    print(f"\n{'=' * 70}")
    print("  LDONet-T -> LDONet-S Knowledge Distillation")
    print(f"{'=' * 70}")
    print(f"  Dataset:      {args.dataset}")
    print(f"  Classes:      {args.num_classes}")
    print(f"  Distill:      {args.distill_method}")
    print(f"  Logit stand:  {args.logit_stand}")
    print(f"  Temperature:  {args.temperature}")
    print(f"  CE weight:    {args.ce_weight}")
    print(f"  KD weight:    {args.kd_weight}")
    if args.distill_method == "dkd":
        print(f"  DKD alpha/beta: {args.dkd_alpha}/{args.dkd_beta}")
    if args.distill_method == "mlkd":
        print(f"  MLKD temps:   {[args.temperature] + list(args.mlkd_temperatures)}")
        print(f"  MLKD cc/bc:   {args.mlkd_cc_weight}/{args.mlkd_bc_weight}")
    if args.distill_method == "ctkd":
        print(f"  CTKD temp range: {args.ctkd_min_temp}/{args.ctkd_max_temp}")
    print(f"  LR:           {args.lr}")
    print(f"  Weight Decay: {args.weight_decay}")
    print(f"  GPU:          {args.gpu_id}")
    print(f"{'=' * 70}\n")

    print("Loading Teacher model: LDONet-T (dual-branch)...")
    teacher = TeacherNet(
        label_num=args.num_classes,
        weight=args.teacher_weight,
    )
    teacher.load_state_dict(torch.load(args.teacher_path, map_location=device))
    teacher.eval().to(device)
    for param in teacher.parameters():
        param.requires_grad = False
    teacher_params = sum(p.numel() for p in teacher.parameters())
    print(f"Teacher loaded from: {args.teacher_path}")
    print(f"Teacher parameters: {teacher_params / 1e6:.2f}M")

    print("\nInitializing Student model: LDONet-S (single-branch)...")
    student = StudentNet(
        label_num=args.num_classes,
        weight=args.student_weight,
    ).to(device)

    if args.student_path:
        student.load_state_dict(torch.load(args.student_path, map_location=device))
        print(f"Student resumed from: {args.student_path}")

    student_params = sum(p.numel() for p in student.parameters())
    print(f"Student parameters: {student_params / 1e6:.2f}M")
    print(f"Compression ratio: {teacher_params / student_params:.2f}x")

    distiller = build_distiller(args.distill_method, student, teacher, cfg).to(device)

    optimizer = optim.AdamW(
        distiller.get_learnable_parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    def lr_lambda(current_epoch):
        # Keep schedule consistent with teacher training for stable optimization.
        if current_epoch < args.warmup_epochs:
            return float(current_epoch + 1) / float(max(1, args.warmup_epochs))

        progress = (current_epoch - args.warmup_epochs) / float(
            max(1, args.epoch_num - args.warmup_epochs)
        )
        cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
        min_factor = args.min_lr / args.lr
        return min_factor + (1.0 - min_factor) * cosine

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    print(f"\n{'=' * 70}")
    print("  Starting Training...")
    print(f"{'=' * 70}\n")

    best_acc = 0.0
    train_losses_ce = []
    train_losses_kd = []
    train_losses_total = []
    train_accuracy = []

    start_time = time.time()

    for epoch in range(args.epoch_num):
        epoch_start = time.time()
        loss_ce, loss_kd, loss_total, accuracy = fit(
            epoch, distiller, data_loader, cfg, optimizer
        )
        epoch_time = time.time() - epoch_start

        scheduler.step()

        train_losses_ce.append(loss_ce)
        train_losses_kd.append(loss_kd)
        train_losses_total.append(loss_total)
        train_accuracy.append(accuracy)

        if epoch % 5 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"epoch {epoch}: lr={current_lr:.8f}, epoch_time={epoch_time:.2f}s")

        if accuracy >= best_acc:
            best_acc = accuracy
            torch.save(student.state_dict(), os.path.join(args.des_path, 'net_params_best.pth'))
            if epoch % 5 == 0:
                print(f"*** New best model saved: accuracy={accuracy:.4f} ***")

        if epoch % args.save_interval == 0 and epoch >= 40:
            torch.save(student.state_dict(),
                       os.path.join(args.des_path, f'epoch_{epoch}_net_params.pth'))

    with open(os.path.join(args.path_rst, 'train_losses_ce.txt'), 'w') as f:
        for v in train_losses_ce:
            f.write(str(v) + '\n')

    with open(os.path.join(args.path_rst, 'train_losses_kd.txt'), 'w') as f:
        for v in train_losses_kd:
            f.write(str(v) + '\n')

    with open(os.path.join(args.path_rst, 'train_losses_total.txt'), 'w') as f:
        for v in train_losses_total:
            f.write(str(v) + '\n')

    with open(os.path.join(args.path_rst, 'train_accuracy.txt'), 'w') as f:
        for v in train_accuracy:
            f.write(str(v) + '\n')

    total_time = time.time() - start_time

    print(f"\n{'=' * 70}")
    print("  Training Complete!")
    print(f"{'=' * 70}")
    print(f"  Best accuracy: {best_acc:.3f}%")
    print(f"  Total time: {total_time / 60:.1f} minutes")
    print(f"  Model saved: {args.des_path}")
    print(f"{'=' * 70}\n")
