"""
Decoupled Knowledge Distillation (DKD).

This module follows the same Distiller interface as distillation.kd.KD so it
can be selected by the training script with a distillation method name.
"""

import torch
import torch.nn.functional as F

from .kd import Distiller, logit_normalize


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    return torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    return torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()


def cat_mask(tensor, mask1, mask2):
    target_part = (tensor * mask1).sum(dim=1, keepdim=True)
    non_target_part = (tensor * mask2).sum(dim=1, keepdim=True)
    return torch.cat([target_part, non_target_part], dim=1)


def dkd_loss(
    logits_student_in,
    logits_teacher_in,
    target,
    alpha,
    beta,
    temperature,
    logit_stand=False,
):
    """DKD loss with optional logit standardization."""
    logits_student = logit_normalize(logits_student_in) if logit_stand else logits_student_in
    logits_teacher = logit_normalize(logits_teacher_in) if logit_stand else logits_teacher_in

    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)

    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student.clamp_min(1e-12))

    tckd_loss = (
        F.kl_div(log_pred_student, pred_teacher, reduction="sum")
        * (temperature ** 2)
        / target.shape[0]
    )

    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask,
        dim=1,
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask,
        dim=1,
    )
    nckd_loss = (
        F.kl_div(log_pred_student_part2, pred_teacher_part2, reduction="sum")
        * (temperature ** 2)
        / target.shape[0]
    )

    return alpha * tckd_loss + beta * nckd_loss, tckd_loss, nckd_loss


class DKD(Distiller):
    """Decoupled KD with optional logit standardization."""

    def __init__(self, student, teacher, cfg):
        super(DKD, self).__init__(student, teacher)
        self.temperature = cfg.TEMPERATURE
        self.ce_loss_weight = cfg.CE_WEIGHT
        self.kd_loss_weight = cfg.KD_WEIGHT
        self.logit_stand = cfg.LOGIT_STAND
        self.alpha = cfg.DKD_ALPHA
        self.beta = cfg.DKD_BETA

    @staticmethod
    def _extract_logits(model_output):
        if isinstance(model_output, tuple):
            return model_output[0]
        return model_output

    def forward_train(self, image, target, **kwargs):
        student_output = self.student(image, target)
        logits_student = self._extract_logits(student_output)

        with torch.no_grad():
            teacher_output = self.teacher(image, None)
            logits_teacher = self._extract_logits(teacher_output)

        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        raw_dkd, loss_tckd, loss_nckd = dkd_loss(
            logits_student,
            logits_teacher,
            target,
            self.alpha,
            self.beta,
            self.temperature,
            self.logit_stand,
        )
        loss_kd = self.kd_loss_weight * raw_dkd

        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
            "loss_tckd": loss_tckd.detach(),
            "loss_nckd": loss_nckd.detach(),
        }
        return logits_student, losses_dict
