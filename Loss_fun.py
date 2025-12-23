import torch.nn.functional as F
from loss.softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from loss.triplet_loss import TripletLoss
from loss.center_loss import CenterLoss

# def make_loss(num_classes):

#     feat_dim =768
#     center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
#     center_criterion2 = CenterLoss(num_classes=num_classes, feat_dim=768*6, use_gpu=True)

#     triplet = TripletLoss()
#     xent = CrossEntropyLabelSmooth(num_classes=num_classes)


#     def loss_func(score, feat, target, target_cam):
#         if isinstance(score, list):
#                 ID_LOSS = [xent(scor, target) for scor in score[1:]]
#                 ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
#                 ID_LOSS = 0.25 * ID_LOSS + 0.75 * xent(score[0], target)
#         else:
#                 ID_LOSS = xent(score, target)

#         if isinstance(feat, list):
#                 TRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:]]
#                 TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
#                 TRI_LOSS = 0.25 * TRI_LOSS + 0.75 * triplet(feat[0], target)[0]

#                 center=center_criterion(feat[0], target)
#                 centr2 = [center_criterion2(feats, target) for feats in feat[1:]]
#                 centr2 = sum(centr2) / len(centr2)
#                 center=0.25 *centr2 +  0.75 *  center
#         else:
#                 TRI_LOSS = triplet(feat, target)[0]

#         print(f"ID_LOSS:{ID_LOSS}, TRI_LOSS:{TRI_LOSS}")
#         # return   ID_LOSS+ TRI_LOSS, center
#         return   ID_LOSS+ TRI_LOSS

#     return  loss_func,center_criterion


import torch.nn.functional as F
from loss.softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from loss.triplet_loss import TripletLoss
from loss.center_loss import CenterLoss


def make_loss(num_classes):
    feat_dim = 768
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
    center_criterion2 = CenterLoss(num_classes=num_classes, feat_dim=4608, use_gpu=True)

    triplet = TripletLoss()
    xent = CrossEntropyLabelSmooth(num_classes=num_classes)

    def loss_func(score, feat, target, target_cam, feat_teacher=None, cross_video_teacher=None,
                  cross_video_student=None, score_teacher=None, masks=None):
        if isinstance(score, list):
            ID_LOSS = [xent(scor, target) for scor in score[1:]]
            ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
            ID_LOSS = 0.25 * ID_LOSS + 0.75 * xent(score[0], target)
        else:
            ID_LOSS = xent(score, target)

        if isinstance(feat, list):
            TRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:]]
            TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
            TRI_LOSS = 0.25 * TRI_LOSS + 0.75 * triplet(feat[0], target)[0]

            center = center_criterion(feat[0], target)
            centr2 = [center_criterion2(feats, target) for feats in feat[1:]]
            centr2 = sum(centr2) / len(centr2)
            center = 0.25 * centr2 + 0.75 * center
        else:
            TRI_LOSS = triplet(feat, target)[0]

        temperature = 4.0
        loss_student = compute_distillation_loss(cross_video_student, cross_video_teacher, masks,
                                                 temperature=temperature)
        # loss_final = compute_final_distillation_loss(feat_teacher,feat, temperature=temperature)

        score_teacher = F.normalize(score_teacher, p=2, dim=-1)
        score = F.normalize(score, p=2, dim=-1)

        soft_teacher = F.softmax(score_teacher.detach() / temperature, dim=-1)
        log_student = F.log_softmax(score / temperature, dim=-1)
        loss_final = F.kl_div(log_student, soft_teacher, reduction='batchmean') * (temperature ** 2)
        loss_final = F.mse_loss(feat, feat_teacher.detach()) * (temperature ** 2)

        print(
            f"ID_LOSS:{ID_LOSS:.8f}, TRI_LOSS:{TRI_LOSS:.8f}, LOSS_STU:{loss_student:.8f}, LOSS_FINAL:{loss_final:.8f}")
        # print(f"ID_LOSS:{ID_LOSS:.8f}, TRI_LOSS:{TRI_LOSS:.8f}, LOSS_FINAL:{loss_final:.8f}")

        # print(
        #     f"ID_LOSS:{ID_LOSS:.8f}, TRI_LOSS:{TRI_LOSS:.8f}, LOSS_STU:{loss_student:.8f}")

        # # return ID_LOSS + TRI_LOSS + loss_student + loss_final
        return ID_LOSS + TRI_LOSS + loss_student + loss_final

        # return ID_LOSS + TRI_LOSS + loss_student

    return loss_func, center_criterion


# def compute_distillation_loss(student_list, teacher_list, masks, temperature=3.0):
#     """
#     计算列表间的蒸馏损失（带温度缩放）

#     参数:
#         student_list: 学生模型的输出列表，每个元素是Tensor
#         teacher_list: 教师模型的输出列表，每个元素是Tensor
#         temperature: 温度参数，控制分布平滑程度
#     """
#     total_loss = 0.0
#     for s, t, w in zip(student_list, teacher_list, masks):
#         # print("s.shape, t.shape", s.shape, t.shape)
#         # 确保张量形状匹配
#         assert s.shape == t.shape, "学生和教师张量形状不匹配"
#         s = F.normalize(s, p=2, dim=-1)
#         t = F.normalize(t, p=2, dim=-1)
#         # 分离教师梯度 + 温度软化
#         soft_teacher = F.softmax(t.detach() / temperature, dim=-1)
#         # 学生输出 + 温度调整
#         log_student = F.log_softmax(s / temperature, dim=-1)
#         kl_loss_per_token = F.kl_div(
#             log_student,
#             soft_teacher,
#             reduction='none'
#         ).sum(dim=-1)  # [B,L]

#         # print("log_student.shape", log_student.shape)
#         # print("kl_loss_per_token.shape", kl_loss_per_token.shape)
#         # print("w.shape", w.shape)

#         b, l, c = w.shape
#         w = w.view(b, l)
#         # print(w)
#         distill_loss = (kl_loss_per_token * w).mean() * (temperature ** 2)
#         total_loss += distill_loss
#         # 返回平均损失（或总损失，按需调整）
#     return total_loss / len(student_list)  # 平均损失


#     # return total_loss                     # 总损失

def compute_distillation_loss(student_list, teacher_list, masks, temperature=3.0):
    """
    计算列表间的蒸馏损失（带温度缩放），使用MSE损失函数

    参数:
        student_list: 学生模型的输出列表，每个元素是Tensor
        teacher_list: 教师模型的输出列表，每个元素是Tensor
        temperature: 温度参数，控制分布平滑程度
    """
    total_loss = 0.0
    for s, t in zip(student_list, teacher_list):
        # 确保张量形状匹配
        assert s.shape == t.shape, "学生和教师张量形状不匹配"

        # 分离教师梯度 + 温度软化
        # soft_teacher = F.softmax(t.detach() / temperature, dim=-1)

        # 学生输出 + 温度软化（注意：MSE直接使用概率值，无需log_softmax）
        # soft_student = F.softmax(s / temperature, dim=-1)

        # 计算MSE损失（直接比较软化后的概率分布）
        loss = F.l1_loss(s, t) * (temperature ** 2)
        total_loss += loss

    # 返回平均损失
    return total_loss / len(student_list)


import torch
import torch.nn.functional as F


def cmca_loss_multi_layer(teacher_feats, student_feats, pids, tau=0.07, layer_weights=None):
    """
    支持多层特征的跨模态对比对齐损失
    Args:
        teacher_feats: List of [B, L, D] 或 [B, D] (冻结的多层教师特征)
        student_feats: List of [B, L, D] 或 [B, D] (可训练的学生多层特征)
        pids: [B] (身份标签)
        tau: 温度系数
        layer_weights: List[float] 各层权重，若为None则平均加权
    """
    # 统一输入格式为列表
    teacher_feats = [teacher_feats] if not isinstance(teacher_feats, list) else teacher_feats
    student_feats = [student_feats] if not isinstance(student_feats, list) else student_feats
    assert len(teacher_feats) == len(student_feats), "Teacher和Student特征层数需一致"

    # 初始化层权重
    num_layers = len(teacher_feats)
    if layer_weights is None:
        layer_weights = [1.0 / num_layers] * num_layers
    else:
        assert len(layer_weights) == num_layers, "层权重数需与特征层数一致"
        layer_weights = [w / sum(layer_weights) for w in layer_weights]  # 归一化

    total_loss = 0.0

    # 逐层计算损失
    for layer_idx in range(num_layers):
        # 获取当前层特征
        t_feat = teacher_feats[layer_idx]  # [B, L, D] 或 [B, D]
        s_feat = student_feats[layer_idx]

        # 若特征为[B, L, D]，展平为[B*L, D]后计算损失
        if t_feat.dim() == 3:
            B, L, D = t_feat.shape
            t_feat = t_feat.mean(1).view(B // 6, 6, D).contiguous().mean(1)
            s_feat = s_feat.mean(1).view(B // 6, 6, D).contiguous().mean(1)
            expanded_pids = pids  # [B*L]
        else:
            expanded_pids = pids  # 原始pids [B]

        # 计算单层损失
        layer_loss = _single_layer_cmca_loss(t_feat, s_feat, expanded_pids, tau)
        total_loss += layer_weights[layer_idx] * layer_loss

    return total_loss


def _single_layer_cmca_loss(teacher_feat, student_feat, pids, tau):
    """
    单层特征对比损失计算
    teacher_feat: [N, D]
    student_feat: [N, D]
    pids: [N]
    """
    # L2归一化
    teacher_feat = F.normalize(teacher_feat, p=2, dim=1)
    student_feat = F.normalize(student_feat, p=2, dim=1)

    N = teacher_feat.size(0)
    device = teacher_feat.device
    loss = 0.0

    # 向量化计算相似度矩阵
    sim_st2te = torch.mm(student_feat, teacher_feat.T)  # [N, N]
    sim_st2st = torch.mm(student_feat, student_feat.T)  # [N, N]

    # 创建标签掩码
    pid_mask = (pids.unsqueeze(1) == pids.unsqueeze(0)).float().to(device)  # [N, N]

    for i in range(N):
        # ----------------------
        # 正样本对
        # ----------------------
        # 跨模态正样本：Student_i 与 Teacher_i
        pos_cross = torch.exp(sim_st2te[i, i] / tau)

        # 模态内正样本：同ID的其他Student特征（排除自身）
        pos_intra = torch.sum(torch.exp(sim_st2st[i, :] / tau) * (pid_mask[i] - torch.eye(N)[i].to(device)))

        # 分子 = 跨模态正样本 + 模态内正样本
        numerator = pos_cross + pos_intra

        # ----------------------
        # 负样本对
        # ----------------------
        # 跨模态负样本：不同ID的Teacher锚点
        neg_cross = torch.sum(torch.exp(sim_st2te[i, :] / tau) * (1 - pid_mask[i]))

        # 模态内负样本：不同ID的Student特征
        neg_intra = torch.sum(torch.exp(sim_st2st[i, :] / tau) * (1 - pid_mask[i]))

        # 分母 = 跨模态负样本 + 模态内负样本 + 分子
        denominator = neg_cross + neg_intra + numerator + 1e-8  # 防止除零

        # 累计损失
        loss += -torch.log(numerator / denominator)

    return loss / N


def compute_final_distillation_loss(feat_teacher_list, feat_list, temperature):
    """
    计算多层级蒸馏损失
    参数:
        feat_teacher_list: 教师模型特征张量列表[T1, T2, ...]
        feat_list: 学生模型特征张量列表[S1, S2, ...]
        temperature: 温度系数
    返回:
        aggregated_loss: 聚合后的蒸馏损失
    """
    total_loss = 0.0
    num_layers = len(feat_teacher_list)

    for feat_teacher, feat in zip(feat_teacher_list, feat_list):
        # 教师特征处理 (分离梯度+softmax)
        soft_teacher = F.softmax(feat_teacher.detach() / temperature, dim=-1)

        # 学生特征处理 (log_softmax)
        log_student = F.log_softmax(feat / temperature, dim=-1)

        # 计算单层KL散度 (batchmean模式自动处理不同形状)
        layer_loss = F.kl_div(
            input=log_student,
            target=soft_teacher,
            reduction='batchmean'
        ) * (temperature ** 2)

        total_loss += layer_loss

    # 按层级数平均损失
    return total_loss / num_layers
