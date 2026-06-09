"""LDONet-S: Student network (paper naming)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .component.SRF import EDTM
from .LDONet_T import GaborConv2d, ArcMarginProduct


class ECA_Module(nn.Module):
    """ECA (Efficient Channel Attention) Module."""

    def __init__(self, channels, gamma=2, b=1):
        super(ECA_Module, self).__init__()
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k_size = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class LDONetSSingleScaleExtractor(torch.nn.Module):
    """Single-scale local extractor (EDTM) for LDONet-S."""

    def __init__(self, channel_in=1, filter_num=24, deg_dim=48, local_dim=24):
        super(LDONetSSingleScaleExtractor, self).__init__()

        self.gabor_large = GaborConv2d(
            channel_in=channel_in,
            channel_out=filter_num,
            kernel_size=13,
            stride=2,
            padding=13 // 2,
            init_ratio=0.4,
        )
        self.ca_large = ECA_Module(filter_num)
        self.downsample = torch.nn.Conv2d(filter_num, filter_num, kernel_size=3, stride=2, padding=1)

        self.conv_0 = torch.nn.Conv2d(filter_num, deg_dim, kernel_size=5, padding=0)
        self.conv_local = torch.nn.Conv2d(deg_dim, local_dim, kernel_size=3, stride=2, padding=0)
        self.max_pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

    def process_block(self, x, conv):
        x = conv(x)
        x = F.relu(x)
        x = self.max_pool(x)
        return x

    def forward(self, x):
        large_scale = self.gabor_large(x)
        large_scale = self.ca_large(large_scale)
        large_scale = self.downsample(large_scale)

        processed = self.process_block(large_scale, self.conv_0)
        local_map = self.conv_local(processed)
        local_feat = local_map.reshape(local_map.size(0), -1)

        return local_feat, processed


class LMGFFM(nn.Module):
    """Lightweight 2-scale SRF-style fusion (Lite MGFFM) for LDONet-S."""

    def __init__(self, in_dim, mid_dim=8, out_size=(30, 30)):
        super(LMGFFM, self).__init__()
        self.out_size = out_size

        self.proj_x1 = nn.Conv2d(in_dim, mid_dim, kernel_size=1, bias=False)
        self.proj_x2 = nn.Conv2d(in_dim, mid_dim, kernel_size=1, bias=False)

        self.gate = nn.Sequential(
            nn.Conv2d(mid_dim, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(mid_dim * 2, mid_dim * 2, kernel_size=3, padding=1, groups=mid_dim * 2, bias=False),
            nn.Conv2d(mid_dim * 2, mid_dim, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_dim, 1, kernel_size=1),
        )

    def forward(self, x1, x2):
        x1 = self.proj_x1(x1)
        x2 = self.proj_x2(x2)

        x2 = F.interpolate(x2, size=x1.shape[-2:], mode='bilinear', align_corners=False)
        x2 = x2 * self.gate(x1)

        x = torch.cat([x1, x2], dim=1)
        x = self.fuse(x)
        if x.shape[-2:] != self.out_size:
            x = F.interpolate(x, size=self.out_size, mode='bilinear', align_corners=False)
        return x


class LDONet_S(torch.nn.Module):
    """LDONet Student: single-scale single-branch network with EDTM + Lite MGFFM."""

    def __init__(self, label_num, weight=0.7):
        super(LDONet_S, self).__init__()

        self.label_num = label_num
        self.weight = weight
        self.deg_dim = 48
        self.local_feat_dim = 24 * 6 * 6
        self.embedding_dim = 1024

        self.feature_extraction = LDONetSSingleScaleExtractor(
            channel_in=1,
            filter_num=24,
            deg_dim=self.deg_dim,
            local_dim=24,
        )

        self.degconv_14 = EDTM(in_dim=self.deg_dim, nbins=6, cell_size=(4, 4))
        self.srf_module = LMGFFM(48, 8, (14, 14))
        self.global_pool = torch.nn.AdaptiveAvgPool2d((1, 1))

        self.fully_connection_1 = torch.nn.Linear(self.local_feat_dim, self.embedding_dim)
        self.fully_connection_for_deg = torch.nn.Linear(self.deg_dim + 1, self.embedding_dim)

        self.dropout = torch.nn.Dropout(p=0.5)
        self.arcface = ArcMarginProduct(in_features=self.embedding_dim, out_features=label_num)

    def forward(self, feature_tensor, target=None):
        processed_feature, attention_weights = self.processing(feature_tensor)
        feature_tensor = self.dropout(processed_feature)
        feature_tensor = self.arcface(feature_tensor, target)
        return feature_tensor, F.normalize(feature_tensor, dim=-1), attention_weights

    def get_feature_vector(self, feature_tensor):
        feature_tensor, _ = self.processing(feature_tensor)
        return F.normalize(feature_tensor, p=2, dim=1, eps=1e-8)

    def processing(self, feature_tensor):
        local_feat_vector, first_order = self.feature_extraction(feature_tensor)

        enhanced_14 = self.degconv_14(first_order)

        srf_x2 = F.adaptive_avg_pool2d(enhanced_14, output_size=(7, 7))
        srf_map = self.srf_module(enhanced_14, srf_x2)

        gate_14 = torch.sigmoid(srf_map)
        enhanced_14 = enhanced_14 * gate_14 + enhanced_14

        global_feat_30 = self.global_pool(enhanced_14).flatten(1)
        global_feat_srf = self.global_pool(srf_map).flatten(1)

        global_feat = torch.cat([global_feat_30, global_feat_srf], dim=1)
        global_feat = self.fully_connection_for_deg(global_feat)

        local_feat = self.fully_connection_1(local_feat_vector)

        fused_feat = local_feat * self.weight + global_feat * (1 - self.weight)
        attention_weights = F.adaptive_avg_pool2d(gate_14, output_size=(12, 12)).flatten(1)

        return fused_feat, attention_weights
