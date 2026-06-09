"""
Curriculum Temperature KD (CTKD) adapted to this project.

This implementation keeps the local Distiller interface and uses a learnable
global temperature parameter. The temperature is clamped to a stable range and
can be optimized together with the student parameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .kd import Distiller, kd_loss


class CTKD(Distiller):
    """Vanilla KD with a learnable curriculum temperature."""

    def __init__(self, student, teacher, cfg):
        super(CTKD, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.CE_WEIGHT
        self.kd_loss_weight = cfg.KD_WEIGHT
        self.logit_stand = cfg.LOGIT_STAND
        self.min_temperature = cfg.CTKD_MIN_TEMP
        self.max_temperature = cfg.CTKD_MAX_TEMP
        initial_temperature = float(cfg.TEMPERATURE)
        self.temperature_logit = nn.Parameter(torch.tensor(initial_temperature).float())

    @staticmethod
    def _extract_logits(model_output):
        if isinstance(model_output, tuple):
            return model_output[0]
        return model_output

    def current_temperature(self):
        return torch.clamp(
            self.temperature_logit,
            min=self.min_temperature,
            max=self.max_temperature,
        )

    def get_learnable_parameters(self):
        return list(self.student.parameters()) + [self.temperature_logit]

    def forward_train(self, image, target, **kwargs):
        student_output = self.student(image, target)
        logits_student = self._extract_logits(student_output)

        with torch.no_grad():
            teacher_output = self.teacher(image, None)
            logits_teacher = self._extract_logits(teacher_output)

        temperature = self.current_temperature()
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_kd = self.kd_loss_weight * kd_loss(
            logits_student,
            logits_teacher,
            temperature,
            self.logit_stand,
        )
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
            "temperature": temperature.detach(),
        }
        return logits_student, losses_dict
