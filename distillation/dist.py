"""
DIST: Knowledge Distillation from A Stronger Teacher.

This implementation ports the logit-relation loss from hunto/DIST_KD to the
local Distiller interface. It distills inter-class and intra-class Pearson
relations between softened teacher and student predictions.
"""

import torch
import torch.nn.functional as F

from .kd import Distiller, logit_normalize


def cosine_similarity(a, b, eps=1e-8):
    return (a * b).sum(dim=1) / (a.norm(dim=1) * b.norm(dim=1) + eps)


def pearson_correlation(a, b, eps=1e-8):
    a = a - a.mean(dim=1, keepdim=True)
    b = b - b.mean(dim=1, keepdim=True)
    return cosine_similarity(a, b, eps)


def inter_class_relation(y_student, y_teacher):
    return 1.0 - pearson_correlation(y_student, y_teacher).mean()


def intra_class_relation(y_student, y_teacher):
    return inter_class_relation(y_student.transpose(0, 1), y_teacher.transpose(0, 1))


def dist_loss(
    logits_student_in,
    logits_teacher_in,
    beta=1.0,
    gamma=1.0,
    temperature=1.0,
    logit_stand=False,
    
):
    logits_student = logit_normalize(logits_student_in) if logit_stand else logits_student_in
    logits_teacher = logit_normalize(logits_teacher_in) if logit_stand else logits_teacher_in

    y_student = F.softmax(logits_student / temperature, dim=1)
    y_teacher = F.softmax(logits_teacher / temperature, dim=1)

    inter_loss = inter_class_relation(y_student, y_teacher)
    intra_loss = intra_class_relation(y_student, y_teacher)
    loss = (beta * inter_loss + gamma * intra_loss) * (temperature ** 2)
    return loss, inter_loss, intra_loss


class DIST(Distiller):
    """DIST logit-relation distiller."""

    def __init__(self, student, teacher, cfg):
        super(DIST, self).__init__(student, teacher)
        self.temperature = cfg.TEMPERATURE
        self.ce_loss_weight = cfg.CE_WEIGHT
        self.kd_loss_weight = cfg.KD_WEIGHT
        self.logit_stand = cfg.LOGIT_STAND
        self.beta = cfg.DIST_BETA
        self.gamma = cfg.DIST_GAMMA

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
        raw_dist, loss_inter, loss_intra = dist_loss(
            logits_student,
            logits_teacher,
            self.beta,
            self.gamma,
            self.temperature,
            self.logit_stand,
        )
        loss_kd = self.kd_loss_weight * raw_dist

        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
            "loss_dist_inter": loss_inter.detach(),
            "loss_dist_intra": loss_intra.detach(),
        }
        return logits_student, losses_dict
