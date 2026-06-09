"""
Knowledge Distillation via Normalized Direction (NDKD).

This adapts DirectNormLoss from:
https://github.com/WangYZ1608/Knowledge-Distillation-via-ND

The original implementation uses precomputed teacher class-mean embeddings.
For this project, the default center source is the teacher ArcFace classifier
weight, which acts as a class prototype and avoids an extra offline pass.
"""

import torch
import torch.nn.functional as F
import json
import numpy as np

from .kd import Distiller, kd_loss


def _extract_logits(model_output):
    if isinstance(model_output, tuple):
        return model_output[0]
    return model_output


def direct_norm_loss(student_features, teacher_features, teacher_centers, target, eps=1e-8):
    """
    Match student feature direction to the teacher class center direction.

    For each sample i with label y:
        1 - dot(s_i, normalize(center_y)) / max(||s_i||, ||t_i||)
    """
    if student_features.shape != teacher_features.shape:
        raise ValueError(
            "NDKD requires student and teacher feature dimensions to match, "
            f"got {tuple(student_features.shape)} and {tuple(teacher_features.shape)}."
        )

    centers = teacher_centers.index_select(0, target.reshape(-1)).detach()
    center_dirs = F.normalize(centers, p=2, dim=1, eps=eps)
    student_norm = student_features.norm(p=2, dim=1).clamp_min(eps)
    teacher_norm = teacher_features.norm(p=2, dim=1).clamp_min(eps)
    max_norm = torch.maximum(student_norm, teacher_norm)
    direction_score = (student_features * center_dirs).sum(dim=1) / max_norm
    return (1.0 - direction_score).mean()


class NDKD(Distiller):
    """CE + logit KD + normalized-direction feature distillation."""

    def __init__(self, student, teacher, cfg):
        super(NDKD, self).__init__(student, teacher)
        self.temperature = cfg.TEMPERATURE
        self.ce_loss_weight = cfg.CE_WEIGHT
        self.kd_loss_weight = cfg.KD_WEIGHT
        self.nd_loss_weight = getattr(cfg, "NDKD_WEIGHT", 1.0)
        self.logit_stand = cfg.LOGIT_STAND
        self.warmup_epochs = getattr(cfg, "WARMUP_EPOCHS", 20)
        self.center_source = getattr(cfg, "NDKD_CENTER_SOURCE", "arcface")
        self.center_path = getattr(cfg, "NDKD_CENTER_PATH", "")
        if self.center_source not in {"arcface", "file"}:
            raise ValueError("NDKD_CENTER_SOURCE must be 'arcface' or 'file'.")
        if self.center_source == "arcface" and (
            not hasattr(self.teacher, "arcface") or not hasattr(self.teacher.arcface, "weight")
        ):
            raise AttributeError("NDKD requires teacher.arcface.weight as class centers.")
        center_tensor = None
        if self.center_source == "file":
            center_tensor = self._load_centers(self.center_path)
        self.register_buffer("file_centers", center_tensor)

    @staticmethod
    def _load_centers(path):
        if not path:
            raise ValueError("NDKD_CENTER_PATH is required when NDKD_CENTER_SOURCE='file'.")
        if path.endswith((".pt", ".pth")):
            centers = torch.load(path, map_location="cpu")
        elif path.endswith(".npy"):
            centers = torch.from_numpy(np.load(path))
        elif path.endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                centers = json.load(f)
            if isinstance(centers, dict):
                centers = [centers[str(index)] for index in range(len(centers))]
            centers = torch.tensor(centers)
        else:
            raise ValueError("NDKD center file must be .pt, .pth, .npy, or .json.")
        if isinstance(centers, dict):
            centers = centers.get("centers", centers.get("teacher_centers"))
        centers = torch.as_tensor(centers, dtype=torch.float32)
        if centers.ndim != 2:
            raise ValueError(f"NDKD centers must be a 2D tensor, got shape {tuple(centers.shape)}.")
        return centers

    def _teacher_centers(self):
        if self.center_source == "file":
            return self.file_centers
        return self.teacher.arcface.weight

    def forward_train(self, image, target, **kwargs):
        student_features, _ = self.student.processing(image)
        logits_student = self.student.arcface(self.student.dropout(student_features), target)

        with torch.no_grad():
            teacher_features, _ = self.teacher.processing(image)
            logits_teacher = self.teacher.arcface(teacher_features, None)

        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_kd = self.kd_loss_weight * kd_loss(
            logits_student,
            logits_teacher,
            self.temperature,
            self.logit_stand,
        )
        loss_nd = self.nd_loss_weight * direct_norm_loss(
            student_features,
            teacher_features,
            self._teacher_centers(),
            target,
        )

        warmup_epochs = max(1, int(self.warmup_epochs))
        warmup_factor = min(kwargs.get("epoch", 0) / warmup_epochs, 1.0)
        loss_total = loss_ce + warmup_factor * (loss_kd + loss_nd)

        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
            "loss_nd": loss_nd,
            "loss_total": loss_total,
        }
        return logits_student, losses_dict
