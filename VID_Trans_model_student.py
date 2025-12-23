import torch
import torch.nn as nn
import copy
from vit_ID_student import TransReID, Block
from functools import partial
from torch.nn import functional as F


def TCSS(features, shift, b, t):
    # aggregate features at patch level
    features = features.view(b, features.size(1), t * features.size(2))
    token = features[:, 0:1]

    batchsize = features.size(0)
    dim = features.size(-1)

    # shift the patches with amount=shift
    features = torch.cat([features[:, shift:], features[:, 1:shift]], dim=1)

    # Patch Shuffling by 2 part
    try:
        features = features.view(batchsize, 2, -1, dim)
    except:
        features = torch.cat([features, features[:, -2:-1, :]], dim=1)
        features = features.view(batchsize, 2, -1, dim)

    features = torch.transpose(features, 1, 2).contiguous()
    features = features.view(batchsize, -1, dim)

    return features, token


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class VID_Trans_student(nn.Module):
    def __init__(self, num_classes, camera_num, pretrainpath):
        super(VID_Trans_student, self).__init__()
        self.in_planes = 768
        self.num_classes = num_classes

        self.base = TransReID(
            img_size=[256, 128], patch_size=16, stride_size=[12, 12], embed_dim=768, depth=12, num_heads=12,
            mlp_ratio=4, qkv_bias=True, \
            camera=camera_num, drop_path_rate=0.1, drop_rate=0.0, attn_drop_rate=0.0,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), cam_lambda=3.0)

        state_dict = torch.load(pretrainpath, map_location='cpu')
        self.base.load_param(state_dict, load=True)

        # global stream
        block = self.base.blocks[-1]
        layer_norm = self.base.norm
        self.b1 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        # -----------------------------------------------
        # -----------------------------------------------

        # building local video stream
        #         dpr = [x.item() for x in torch.linspace(0, 0, 12)]  # stochastic depth decay rule

        #         self.block1 = Block(
        #                 dim=768 * 4, num_heads=12, mlp_ratio=4, qkv_bias=True, qk_scale=None,
        #                 drop=0, attn_drop=0, drop_path=dpr[11], norm_layer=partial(nn.LayerNorm, eps=1e-6))

        #         self.b2 = nn.Sequential(
        #             self.block1,
        #             nn.LayerNorm(768 * 4)#copy.deepcopy(layer_norm)
        #         )

        #         self.bottleneck_1 = nn.BatchNorm1d(768 * 4)
        #         self.bottleneck_1.bias.requires_grad_(False)
        #         self.bottleneck_1.apply(weights_init_kaiming)
        #         self.bottleneck_2 = nn.BatchNorm1d(768 * 4)
        #         self.bottleneck_2.bias.requires_grad_(False)
        #         self.bottleneck_2.apply(weights_init_kaiming)
        #         self.bottleneck_3 = nn.BatchNorm1d(768 * 4)
        #         self.bottleneck_3.bias.requires_grad_(False)
        #         self.bottleneck_3.apply(weights_init_kaiming)
        #         self.bottleneck_4 = nn.BatchNorm1d(768 * 4)
        #         self.bottleneck_4.bias.requires_grad_(False)
        #         self.bottleneck_4.apply(weights_init_kaiming)

        #         self.classifier_1 = nn.Linear(768 * 4, self.num_classes, bias=False)
        #         self.classifier_1.apply(weights_init_classifier)
        #         self.classifier_2 = nn.Linear(768 * 4, self.num_classes, bias=False)
        #         self.classifier_2.apply(weights_init_classifier)
        #         self.classifier_3 = nn.Linear(768 * 4, self.num_classes, bias=False)
        #         self.classifier_3.apply(weights_init_classifier)
        #         self.classifier_4 = nn.Linear(768 * 4, self.num_classes, bias=False)
        #         self.classifier_4.apply(weights_init_classifier)

        # -------------------video attention-------------
        # self.middle_dim = 256  # middle layer dimension
        # self.attention_conv = nn.Conv2d(self.in_planes, self.middle_dim,
        #                                 [1, 1])  # 7,4 cooresponds to 224, 112 input image size
        # self.attention_tconv = nn.Conv1d(self.middle_dim, 1, 3, padding=1)
        # self.attention_conv.apply(weights_init_kaiming)
        # self.attention_tconv.apply(weights_init_kaiming)
        # ------------------------------------------

        self.shift_num = 5
        self.part = 4
        self.rearrange = True
        self.mask_gene = MultiMaskGenerator(num_generators=2)

    def forward(self, x, y=None, label=None, cam_label=None, view_label=None,
                cross_video_teacher=None):  # label is unused if self.cos_layer == 'no'

        b = x.size(0)
        t = x.size(1)

        x = x.reshape(x.size(0) * x.size(1), x.size(2), x.size(3), x.size(4))
        if y is not None:
            B, T, L, C1 = y.shape
            y = y.reshape(-1, L, C1)

        features, cross_video, _, _ = self.base(x, y=y, cam_label=cam_label)

        # global branch
        b1_feat = self.b1(features)  # [64, 129, 768]
        global_feat = b1_feat[:, 0]

#         global_feat = global_feat.unsqueeze(dim=2)
#         global_feat = global_feat.unsqueeze(dim=3)
#         a = F.relu(self.attention_conv(global_feat))
#         a = a.view(b, t, self.middle_dim)
#         a = a.permute(0, 2, 1)
#         a = F.relu(self.attention_tconv(a))
#         a = a.view(b, t)
#         a_vals = a

#         a = F.softmax(a, dim=1)
#         x = global_feat.view(b, t, -1)
#         a = torch.unsqueeze(a, -1)
#         a = a.expand(b, t, self.in_planes)
#         att_x = torch.mul(x, a)
#         att_x = torch.sum(att_x, 1)

#         global_feat = att_x.view(b, self.in_planes)

        global_feat = global_feat.view(b, t, -1).mean(1)
        feat = self.bottleneck(global_feat)

        # -------------------------------------------------
        # -------------------------------------------------

        # video patch patr features

        #         feature_length = features.size(1) - 1
        #         patch_length = feature_length // 4

        #         #Temporal clip shift and shuffled
        #         x ,token=TCSS(features, self.shift_num, b,t)

        #         # part1
        #         part1 = x[:, :patch_length]
        #         part1 = self.b2(torch.cat((token, part1), dim=1))
        #         part1_f = part1[:, 0]

        #         # part2
        #         part2 = x[:, patch_length:patch_length*2]
        #         part2 = self.b2(torch.cat((token, part2), dim=1))
        #         part2_f = part2[:, 0]

        #         # part3
        #         part3 = x[:, patch_length*2:patch_length*3]
        #         part3 = self.b2(torch.cat((token, part3), dim=1))
        #         part3_f = part3[:, 0]

        #         # part4
        #         part4 = x[:, patch_length*3:patch_length*4]
        #         part4 = self.b2(torch.cat((token, part4), dim=1))
        #         part4_f = part4[:, 0]

        #         part1_bn = self.bottleneck_1(part1_f)
        #         part2_bn = self.bottleneck_2(part2_f)
        #         part3_bn = self.bottleneck_3(part3_f)
        #         part4_bn = self.bottleneck_4(part4_f)

        if self.training:
            
            temp = [torch.cat([student, teacher.detach()], dim=-1)
                for student, teacher in zip(cross_video, cross_video_teacher)]
            masks = self.mask_gene(temp)
            
            Global_ID = self.classifier(feat)
            # Local_ID1 = self.classifier_1(part1_bn)
            # Local_ID2 = self.classifier_2(part2_bn)
            # Local_ID3 = self.classifier_3(part3_bn)
            # Local_ID4 = self.classifier_4(part4_bn)

            # return [Global_ID, Local_ID1, Local_ID2, Local_ID3, Local_ID4 ], [global_feat, part1_f, part2_f, part3_f,part4_f],  a_vals #[global_feat, part1_f, part2_f, part3_f,part4_f],  a_vals 
            return Global_ID, global_feat, cross_video, masks

        else:
            # return torch.cat([feat, part1_bn/4 , part2_bn/4 , part3_bn /4, part4_bn/4 ], dim=1)
            return feat

    def load_param(self, trained_path, load=False):
        if not load:
            param_dict = torch.load(trained_path)
            for i in param_dict:
                self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
                print('Loading pretrained model from {}'.format(trained_path))
        else:
            param_dict = trained_path
            for i in param_dict:
                # print(i)
                if i not in self.state_dict() or 'classifier' in i or 'sie_embed' in i:
                    continue
                self.state_dict()[i].copy_(param_dict[i])

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


import torch
import torch.nn as nn


# class MaskGenerator(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.layernorm = nn.LayerNorm(768)  # 通道归一化
#         # 三个 1D 卷积层，保持空间维度不变
#         self.conv1 = nn.Conv1d(768, 512, kernel_size=1)
#         self.conv2 = nn.Conv1d(512, 256, kernel_size=1)
#         self.conv3 = nn.Conv1d(256, 1, kernel_size=1)  # 输出单通道掩码
#         self.prelu = nn.ReLU() 
#     def forward(self, x):
#         """
#         x: 输入特征 [B, L, C] 其中 C=768
#         返回: 掩码 [B, L, 1]
#         """
#         # 1. LayerNorm 归一化
#         x = self.layernorm(x)  # [B, L, 768]

#         # 2. 转置维度用于卷积: [B, L, C] -> [B, C, L]
#         x = x.transpose(1, 2)

#         # 3. 三个卷积层 + PReLU
#         x = self.conv1(x)  # [B, 768, L]
#         x = self.conv2(x)  # [B, 768, L]
#         x = self.prelu(self.conv3(x))  # [B, 1, L]
#         # 4. 转置回原始维度: [B, 1, L] -> [B, L, 1]
#         x = x.transpose(1, 2)
#         return x

class MaskGenerator(nn.Module):
    def __init__(self):
        super().__init__()

        self.layernorm0 = nn.LayerNorm(768 * 2)
        self.layernorm = nn.LayerNorm(768)
        self.layernorm1 = nn.LayerNorm(384)
        # 使用线性层替代部分卷积（更高效）
        self.fc0 = nn.Linear(768 * 2, 768)
        self.fc1 = nn.Linear(768, 384)
        self.fc2 = nn.Linear(384, 1)
        # self.fc3 = nn.Linear(256, 1)  # 输出单通道掩码
        # 使用ReLU6限制输出范围并引入稀疏性
        self.activation = nn.ReLU6()
        # self.activation = nn.ReLU6()
        # 可选的dropout增加稀疏性
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        """
        x: 输入特征 [B, L, C] 其中 C=768
        返回: 掩码 [B, L, 1] 值域 [0, 6]
        """
        # 1. LayerNorm 归一化
        x = self.layernorm0(x)  # [B, L, 768]
        x = self.fc0(x)
        # 2. 线性映射
        x = self.layernorm1(self.fc1(x))  # [B, L, 384]
        x = self.fc2(x)  # [B, L, 1]
        x = self.dropout(x)
        # 5. 使用ReLU6确保非负值 (0-6) 并允许部分为0
        return self.activation(x)


class MultiMaskGenerator(nn.Module):
    def __init__(self, num_generators=5):
        super().__init__()
        # 创建独立的掩码生成器
        self.generators = nn.ModuleList(
            [MaskGenerator() for _ in range(num_generators)]
        )

    def forward(self, features_list):
        """
        features_list: 包含 5 个特征的列表，每个特征形状为 [B, L, 768]
        返回: 包含 5 个掩码的列表，每个掩码形状为 [B, L, 1]
        """
        assert len(features_list) == len(self.generators), \
            "输入特征数量必须与生成器数量匹配"

        masks = []
        for feature, generator in zip(features_list, self.generators):
            masks.append(generator(feature))

        # print(masks[0])
        return masks


# 使用示例
if __name__ == "__main__":
    # 模拟输入数据：5 个特征，每个形状为 [2, 100, 768]
    features = [torch.rand(2, 100, 768) for _ in range(5)]

    # 创建多掩码生成器
    model = MultiMaskGenerator(num_generators=5)

    # 生成掩码
    masks = model(features)

    # 检查输出
    for i, mask in enumerate(masks):
        print(f"掩码 {i + 1} 形状: {mask.shape}")
        print(f"掩码 {i + 1} 值范围: [{mask.min():.4f}, {mask.max():.4f}]")

