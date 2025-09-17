# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision import transforms
#
# from ._base import Distiller
#
# def normalize(logit):
#     mean = logit.mean(dim=-1, keepdims=True)
#     stdv = logit.std(dim=-1, keepdims=True)
#     return (logit - mean) / (1e-7 + stdv)
#
#
# def kd_loss(logits_student_in, logits_teacher_in, temperature, logit_stand):
#     logits_student = normalize(logits_student_in) if logit_stand else logits_student_in
#     logits_teacher = normalize(logits_teacher_in) if logit_stand else logits_teacher_in
#     log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
#     pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
#     loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
#     loss_kd *= temperature**2
#     return loss_kd
#
#
# class KD(Distiller):
#
#     """Distilling the Knowledge in a Neural Network"""
#
#     def __init__(self, student, teacher, cfg):
#         super(KD, self).__init__(student, teacher)
#         self.temperature = cfg.KD.TEMPERATURE
#         self.ce_loss_weight = cfg.KD.LOSS.CE_WEIGHT
#         self.kd_loss_weight = cfg.KD.LOSS.KD_WEIGHT
#         self.logit_stand = cfg.EXPERIMENT.LOGIT_STAND
#
#     def forward_train(self, image, target, **kwargs):
#         device = image.device
#         logits_student, _ = self.student(image)
#
#         with torch.no_grad():
#             logits_teacher, _ = self.teacher(image)
#
#         loss_kd = self.kd_loss_weight * kd_loss(
#             logits_student, logits_teacher, self.temperature, self.logit_stand
#         )
#
#         loss_ce = (self.ce_loss_weight * F.cross_entropy(logits_student, target))
#
#         losses_dict = {
#             "loss_kd": loss_kd,
#             "loss_ce": loss_ce,
#         }
#         return logits_student, losses_dict




# # 只有ce_loss进行困难样本挖掘
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision import transforms
#
# from ._base import Distiller
#
# def normalize(logit):
#     mean = logit.mean(dim=-1, keepdims=True)
#     stdv = logit.std(dim=-1, keepdims=True)
#     return (logit - mean) / (1e-7 + stdv)
#
#
# def kd_loss(logits_student_in, logits_teacher_in, temperature, logit_stand):
#     logits_student = normalize(logits_student_in) if logit_stand else logits_student_in
#     logits_teacher = normalize(logits_teacher_in) if logit_stand else logits_teacher_in
#     log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
#     pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
#     loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
#     loss_kd *= temperature**2
#     return loss_kd
#
# class KD(Distiller):
#     """Distilling the Knowledge in a Neural Network with hard sample mining"""
#
#     def __init__(self, student, teacher, cfg):
#         super(KD, self).__init__(student, teacher)
#         self.temperature = cfg.KD.TEMPERATURE
#         self.ce_loss_weight = cfg.KD.LOSS.CE_WEIGHT
#         self.kd_loss_weight = cfg.KD.LOSS.KD_WEIGHT
#         self.logit_stand = cfg.EXPERIMENT.LOGIT_STAND
#         # 困难样本比例和权重配置
#         self.hard_ratio = 0.30
#         self.easy_weight = 1.0
#         self.hard_weight = 2.0
#
#     def forward_train(self, image, target, **kwargs):
#         device = image.device
#         logits_student, _ = self.student(image)
#
#         with torch.no_grad():
#             logits_teacher, _ = self.teacher(image)
#
#         # 计算KD损失
#         loss_kd = self.kd_loss_weight * kd_loss(
#             logits_student, logits_teacher, self.temperature, self.logit_stand
#         )
#
#         # 计算每个样本的损失，区分困难样本和简单样本
#         # 使用教师模型的logit计算每个样本的CE损失，用于判断困难样本
#         per_sample_loss_teacher = F.cross_entropy(logits_teacher, target, reduction='none')
#
#         # 确定难例和简单样本的索引
#         hard_samples_num = int(self.hard_ratio * len(per_sample_loss_teacher))
#         _, hard_indices = torch.topk(per_sample_loss_teacher, hard_samples_num)
#
#         # 创建掩码区分难例和简单样本
#         is_hard = torch.zeros_like(per_sample_loss_teacher, dtype=torch.bool)
#         is_hard[hard_indices] = True
#         is_easy = ~is_hard
#
#         # 计算学生模型在难例和简单样本上的CE损失
#         per_sample_loss_student = F.cross_entropy(logits_student, target, reduction='none')
#         hard_loss = per_sample_loss_student[is_hard].mean() if is_hard.any() else 0.0
#         easy_loss = per_sample_loss_student[is_easy].mean() if is_easy.any() else 0.0
#
#         # 加权组合交叉熵损失
#         loss_ce = self.ce_loss_weight * (self.hard_weight * hard_loss + self.easy_weight * easy_loss)
#
#         losses_dict = {
#             "loss_kd": loss_kd,
#             "loss_ce": loss_ce,
#         }
#         return logits_student, losses_dict



# 只有ce_loss进行困难样本挖掘，根据学生自己的损失来进行排序
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
    loss_kd *= temperature**2
    return loss_kd

class KD(Distiller):
    """Distilling the Knowledge in a Neural Network with hard sample mining"""

    def __init__(self, student, teacher, cfg):
        super(KD, self).__init__(student, teacher)
        self.temperature = cfg.KD.TEMPERATURE
        self.ce_loss_weight = cfg.KD.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.KD.LOSS.KD_WEIGHT
        self.logit_stand = cfg.EXPERIMENT.LOGIT_STAND
        # 困难样本比例和权重配置
        self.hard_ratio = 0.30
        self.easy_weight = 1.0
        self.hard_weight = 2.0

    def forward_train(self, image, target, **kwargs):
        device = image.device
        logits_student, _ = self.student(image)

        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        # 计算KD损失
        loss_kd = self.kd_loss_weight * kd_loss(
            logits_student, logits_teacher, self.temperature, self.logit_stand
        )

        # 计算每个样本的损失，区分困难样本和简单样本
        # 使用教师模型的logit计算每个样本的CE损失，用于判断困难样本
        per_sample_loss_student = F.cross_entropy(logits_student, target, reduction='none')

        # 确定难例和简单样本的索引
        hard_samples_num = int(self.hard_ratio * len(per_sample_loss_student))
        _, hard_indices = torch.topk(per_sample_loss_student, hard_samples_num)

        # 创建掩码区分难例和简单样本
        is_hard = torch.zeros_like(per_sample_loss_student, dtype=torch.bool)
        is_hard[hard_indices] = True
        is_easy = ~is_hard

        # 计算学生模型在难例和简单样本上的CE损失
        hard_loss = per_sample_loss_student[is_hard].mean() if is_hard.any() else 0.0
        easy_loss = per_sample_loss_student[is_easy].mean() if is_easy.any() else 0.0

        # 加权组合交叉熵损失
        loss_ce = self.ce_loss_weight * (self.hard_weight * hard_loss + self.easy_weight * easy_loss)

        losses_dict = {
            "loss_kd": loss_kd,
            "loss_ce": loss_ce,
        }
        return logits_student, losses_dict

