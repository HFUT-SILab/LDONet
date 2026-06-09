"""
Knowledge Distillation Module with Logit Standardization
Based on: Logit Standardization in Knowledge Distillation (CVPR 2024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Distiller(nn.Module):
    def __init__(self, student, teacher):
        super(Distiller, self).__init__()
        self.student = student
        self.teacher = teacher

    def train(self, mode=True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            module.train(mode)
        self.teacher.eval()
        return self

    def get_learnable_parameters(self):
        return [v for k, v in self.student.named_parameters()]

    def get_extra_parameters(self):
        return 0

    def forward_train(self, **kwargs):
        raise NotImplementedError()

    def forward_test(self, image):
        student_output = self.student(image)
        if isinstance(student_output, tuple):
            return student_output[0]
        return student_output

    def forward(self, **kwargs):
        if self.training:
            return self.forward_train(**kwargs)
        return self.forward_test(kwargs["image"])


def logit_normalize(logit):
    """
    Z-score normalization for logits
    Z(x; τ) = (x - μ) / σ / τ
    
    Args:
        logit: logit tensor of shape (batch_size, num_classes)
    Returns:
        normalized logit with zero mean and unit variance
    """
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    return (logit - mean) / (1e-7 + stdv)


def kd_loss(logits_student_in, logits_teacher_in, temperature, logit_stand):
    """
    Knowledge Distillation Loss with optional logit standardization
    
    Args:
        logits_student_in: student logit tensor
        logits_teacher_in: teacher logit tensor
        temperature: temperature parameter T
        logit_stand: whether to use logit standardization
    Returns:
        KL divergence loss multiplied by temperature^2
    """
    # Apply logit standardization if enabled
    logits_student = logit_normalize(logits_student_in) if logit_stand else logits_student_in
    logits_teacher = logit_normalize(logits_teacher_in) if logit_stand else logits_teacher_in
    
    # Compute KL divergence
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature**2
    return loss_kd


class KD(Distiller):
    """
    Vanilla Knowledge Distillation with Logit Standardization
    Distilling the Knowledge in a Neural Network (Hinton et al., 2015)
    """

    def __init__(self, student, teacher, cfg):
        super(KD, self).__init__(student, teacher)
        self.temperature = cfg.TEMPERATURE
        self.ce_loss_weight = cfg.CE_WEIGHT
        self.kd_loss_weight = cfg.KD_WEIGHT
        self.logit_stand = cfg.LOGIT_STAND

    @staticmethod
    def _extract_logits(model_output):
        """Support models that return logits or (logits, ...)."""
        if isinstance(model_output, tuple):
            return model_output[0]
        return model_output

    def forward_train(self, image, target, **kwargs):
        student_output = self.student(image, target)
        logits_student = self._extract_logits(student_output)
        with torch.no_grad():
            teacher_output = self.teacher(image, None)
            logits_teacher = self._extract_logits(teacher_output)

        # Losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_kd = self.kd_loss_weight * kd_loss(
            logits_student, logits_teacher, self.temperature, self.logit_stand
        )
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
        }
        return logits_student, losses_dict
