"""
Multi-Level Logit Distillation (MLKD) adapted to this project.

The original implementation combines multiple-temperature KD with class-wise
and batch-wise logit relation consistency. This version keeps the same loss
ideas while matching the local Distiller interface.
"""

import torch
import torch.nn.functional as F

from .kd import Distiller, kd_loss, logit_normalize


def cc_loss(logits_student_in, logits_teacher_in, temperature, logit_stand=False):
    logits_student = logit_normalize(logits_student_in) if logit_stand else logits_student_in
    logits_teacher = logit_normalize(logits_teacher_in) if logit_stand else logits_teacher_in
    _, class_num = logits_teacher.shape
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    student_matrix = torch.mm(pred_student.transpose(1, 0), pred_student)
    teacher_matrix = torch.mm(pred_teacher.transpose(1, 0), pred_teacher)
    return ((teacher_matrix - student_matrix) ** 2).sum() / class_num


def bc_loss(logits_student_in, logits_teacher_in, temperature, logit_stand=False):
    logits_student = logit_normalize(logits_student_in) if logit_stand else logits_student_in
    logits_teacher = logit_normalize(logits_teacher_in) if logit_stand else logits_teacher_in
    batch_size, _ = logits_teacher.shape
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    student_matrix = torch.mm(pred_student, pred_student.transpose(1, 0))
    teacher_matrix = torch.mm(pred_teacher, pred_teacher.transpose(1, 0))
    return ((teacher_matrix - student_matrix) ** 2).sum() / batch_size


def mlkd_loss(
    logits_student,
    logits_teacher,
    temperatures,
    logit_stand=False,
    kd_weight=1.0,
    cc_weight=1.0,
    bc_weight=1.0,
):
    loss_kd = logits_student.new_tensor(0.0)
    loss_cc = logits_student.new_tensor(0.0)
    loss_bc = logits_student.new_tensor(0.0)

    for temperature in temperatures:
        loss_kd = loss_kd + kd_loss(
            logits_student,
            logits_teacher,
            temperature,
            logit_stand,
        )
        loss_cc = loss_cc + cc_loss(
            logits_student,
            logits_teacher,
            temperature,
            logit_stand,
        )
        loss_bc = loss_bc + bc_loss(
            logits_student,
            logits_teacher,
            temperature,
            logit_stand,
        )

    return kd_weight * loss_kd + cc_weight * loss_cc + bc_weight * loss_bc, loss_kd, loss_cc, loss_bc


class MLKD(Distiller):
    """MLKD with optional logit standardization."""

    def __init__(self, student, teacher, cfg):
        super(MLKD, self).__init__(student, teacher)
        self.temperature = cfg.TEMPERATURE
        self.ce_loss_weight = cfg.CE_WEIGHT
        self.kd_loss_weight = cfg.KD_WEIGHT
        self.logit_stand = cfg.LOGIT_STAND
        self.extra_temperatures = tuple(cfg.MLKD_TEMPERATURES)
        self.cc_weight = cfg.MLKD_CC_WEIGHT
        self.bc_weight = cfg.MLKD_BC_WEIGHT

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

        temperatures = (self.temperature,) + self.extra_temperatures
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        raw_loss, loss_kd_raw, loss_cc, loss_bc = mlkd_loss(
            logits_student,
            logits_teacher,
            temperatures,
            self.logit_stand,
            kd_weight=1.0,
            cc_weight=self.cc_weight,
            bc_weight=self.bc_weight,
        )
        loss_kd = self.kd_loss_weight * raw_loss

        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
            "loss_mlkd_kd": loss_kd_raw.detach(),
            "loss_mlkd_cc": loss_cc.detach(),
            "loss_mlkd_bc": loss_bc.detach(),
        }
        return logits_student, losses_dict
