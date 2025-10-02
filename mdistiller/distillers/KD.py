import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from ._base import Distiller


def normalize(logit):
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    return (logit - mean) / (1e-7 + stdv)


def kd_loss(logits_student_in, logits_teacher_in, temperature, logit_stand):
    logits_student = normalize(logits_student_in) if logit_stand else logits_student_in
    logits_teacher = normalize(logits_teacher_in) if logit_stand else logits_teacher_in
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature ** 2
    return loss_kd


def compute_weighted_ce_with_hard_mining(logits_student, logits_teacher, target,
                                         hard_ratio, hard_weight, easy_weight, ce_loss_weight):

    per_sample_loss_teacher = F.cross_entropy(logits_teacher, target, reduction='none')

    hard_samples_num = int(hard_ratio * len(per_sample_loss_teacher))
    _, hard_indices = torch.topk(per_sample_loss_teacher, hard_samples_num)

    is_hard = torch.zeros_like(per_sample_loss_teacher, dtype=torch.bool)
    is_hard[hard_indices] = True
    is_easy = ~is_hard

    per_sample_loss_student = F.cross_entropy(logits_student, target, reduction='none')
    hard_loss = per_sample_loss_student[is_hard].mean() if is_hard.any() else 0.0
    easy_loss = per_sample_loss_student[is_easy].mean() if is_easy.any() else 0.0

    return ce_loss_weight * (hard_weight * hard_loss + easy_weight * easy_loss)


class KD(Distiller):
    """Distilling the Knowledge in a Neural Network with hard sample mining"""

    def __init__(self, student, teacher, cfg):
        super(KD, self).__init__(student, teacher)
        self.temperature = cfg.KD.TEMPERATURE
        self.ce_loss_weight = cfg.KD.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.KD.LOSS.KD_WEIGHT
        self.logit_stand = cfg.EXPERIMENT.LOGIT_STAND
        self.hard_ratio = cfg.KD.LOSS.HARD_RATIO
        self.easy_weight = cfg.KD.LOSS.EASY_WEIGHT
        self.hard_weight = cfg.KD.LOSS.HARD_WEIGHT

    def forward_train(self, image, target, **kwargs):
        device = image.device
        logits_student, _ = self.student(image)

        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)


        loss_kd = self.kd_loss_weight * kd_loss(
            logits_student, logits_teacher, self.temperature, self.logit_stand
        )

        loss_ce = compute_weighted_ce_with_hard_mining(
            logits_student, logits_teacher, target,
            self.hard_ratio, self.hard_weight,
            self.easy_weight, self.ce_loss_weight
        )

        losses_dict = {
            "loss_kd": loss_kd,
            "loss_ce": loss_ce,
        }
        return logits_student, losses_dict
