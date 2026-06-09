"""LDONet-T: Teacher network (paper naming)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import Parameter

from .component.SRF import EDTM, MGFFM


# ---------------------------------------------------------------------------
# Learnable Gabor Filter
# ---------------------------------------------------------------------------
class GaborConv2d(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size, stride=1, padding=0, init_ratio=1):
        super(GaborConv2d, self).__init__()
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.init_ratio = init_ratio if init_ratio > 0 else 1.0
        self.kernel = 0

        self._SIGMA = 9.2 * self.init_ratio
        self._FREQ = 0.057 / self.init_ratio
        self._GAMMA = 2.0

        self.gamma = nn.Parameter(torch.FloatTensor([self._GAMMA]), requires_grad=True)
        self.sigma = nn.Parameter(torch.FloatTensor([self._SIGMA]), requires_grad=True)
        self.theta = nn.Parameter(
            torch.FloatTensor(torch.arange(0, channel_out).float()) * math.pi / channel_out,
            requires_grad=False
        )
        self.f = nn.Parameter(torch.FloatTensor([self._FREQ]), requires_grad=True)
        self.psi = nn.Parameter(torch.FloatTensor([0]), requires_grad=False)

    def genGaborBank(self, kernel_size, channel_in, channel_out, sigma, gamma, theta, f, psi):
        xmax = kernel_size // 2
        ymax = kernel_size // 2
        xmin, ymin = -xmax, -ymax
        ksize = xmax - xmin + 1

        y_0 = torch.arange(ymin, ymax + 1).float()
        x_0 = torch.arange(xmin, xmax + 1).float()
        y = y_0.view(1, -1).repeat(channel_out, channel_in, ksize, 1)
        x = x_0.view(-1, 1).repeat(channel_out, channel_in, 1, ksize)
        x = x.float().to(sigma.device)
        y = y.float().to(sigma.device)

        x_theta = x * torch.cos(theta.view(-1, 1, 1, 1)) + y * torch.sin(theta.view(-1, 1, 1, 1))
        y_theta = -x * torch.sin(theta.view(-1, 1, 1, 1)) + y * torch.cos(theta.view(-1, 1, 1, 1))

        gb = -torch.exp(
            -0.5 * ((gamma * x_theta) ** 2 + y_theta ** 2) / (8 * sigma.view(-1, 1, 1, 1) ** 2)
        ) * torch.cos(2 * math.pi * f.view(-1, 1, 1, 1) * x_theta + psi.view(-1, 1, 1, 1))
        gb = gb - gb.mean(dim=[2, 3], keepdim=True)
        return gb

    def forward(self, x):
        kernel = self.genGaborBank(
            self.kernel_size, self.channel_in, self.channel_out,
            self.sigma, self.gamma, self.theta, self.f, self.psi
        )
        self.kernel = kernel
        return F.conv2d(x, kernel, stride=self.stride, padding=self.padding)


# ---------------------------------------------------------------------------
# ArcFace classification head
# ---------------------------------------------------------------------------
class ArcMarginProduct(nn.Module):
    """ArcFace: large margin arc distance classification head."""

    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label=None):
        if self.training:
            assert label is not None
            cosine = F.linear(F.normalize(input), F.normalize(self.weight))
            sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
            phi = cosine * self.cos_m - sine * self.sin_m
            if self.easy_margin:
                phi = torch.where(cosine > 0, phi, cosine)
            else:
                phi = torch.where(cosine > self.th, phi, cosine - self.mm)
            one_hot = torch.zeros(cosine.size(), device=cosine.device)
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
            output *= self.s
        else:
            cosine = F.linear(F.normalize(input), F.normalize(self.weight))
            output = self.s * cosine
        return output


# ---------------------------------------------------------------------------
# ECA (Efficient Channel Attention)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Dual-scale local feature extractor
# ---------------------------------------------------------------------------
class LDONetTDualScaleExtractor(nn.Module):
    """Dual-scale local feature extractor (EDTM backbone) for LDONet-T."""

    def __init__(self, channel_in=1, filter_num=36):
        super(LDONetTDualScaleExtractor, self).__init__()
        self.filter_num = filter_num

        self.gabor_large = GaborConv2d(
            channel_in=channel_in, channel_out=filter_num,
            kernel_size=17, stride=2, padding=17 // 2, init_ratio=0.4
        )
        self.gabor_small = GaborConv2d(
            channel_in=channel_in, channel_out=filter_num,
            kernel_size=9, stride=2, padding=9 // 2, init_ratio=0.25
        )

        self.ca_large = ECA_Module(filter_num)
        self.ca_small = ECA_Module(filter_num)
        self.small_hw_align = nn.Conv2d(filter_num, filter_num, kernel_size=5, stride=2, padding=2)

        self.conv_0 = nn.Conv2d(filter_num, 64, kernel_size=5, padding=0)
        self.conv_1 = nn.Conv2d(filter_num, 64, kernel_size=5, padding=0)
        self.conv_2 = nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=0)
        self.conv_3 = nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=0)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.suppression_alpha_small = 0.8
        self.suppression_sigmoid_k_small = 18.0
        self.suppression_eps = 1e-6
        self.smoothing_kernel_small = 3

    def process_block(self, x, conv):
        x = conv(x)
        x = F.relu(x)
        x = self.max_pool(x)
        return x

    def _build_suppression_mask(self, large_scale):
        channel_mean = large_scale.mean(dim=(2, 3), keepdim=True)
        channel_std = large_scale.std(dim=(2, 3), keepdim=True, unbiased=False)
        standardized = (channel_mean - large_scale) / (channel_std + self.suppression_eps)
        return torch.sigmoid(self.suppression_sigmoid_k_small * standardized)

    def _decouple_small_scale(self, small_scale, valley_mask):
        padding = self.smoothing_kernel_small // 2
        local_baseline = F.avg_pool2d(
            small_scale, kernel_size=self.smoothing_kernel_small,
            stride=1, padding=padding,
        )
        return small_scale + self.suppression_alpha_small * valley_mask * (local_baseline - small_scale)

    def forward(self, x):
        large_scale = self.gabor_large(x)
        small_scale = self.gabor_small(x)

        valley_mask = self._build_suppression_mask(large_scale)
        small_scale = self._decouple_small_scale(small_scale, valley_mask)

        large_scale = self.ca_large(large_scale)
        small_scale = self.ca_small(small_scale)
        small_scale = self.small_hw_align(small_scale)

        large_processed = self.process_block(large_scale, self.conv_0)
        small_processed = self.process_block(small_scale, self.conv_1)

        large_local = self.conv_2(large_processed)
        small_local = self.conv_3(small_processed)

        local_feat = torch.cat([
            large_local.reshape(large_local.size(0), -1),
            small_local.reshape(small_local.size(0), -1),
        ], dim=1)

        return local_feat, large_processed, small_processed


# ---------------------------------------------------------------------------
# LDONet-T
# ---------------------------------------------------------------------------
class LDONet_T(torch.nn.Module):
    """LDONet Teacher: dual-scale dual-branch network with EDTM + MGFFM."""

    def __init__(self, label_num, weight=0.7):
        super(LDONet_T, self).__init__()
        self.label_num = label_num
        self.weight = weight

        self.feature_extraction = LDONetTDualScaleExtractor(
            channel_in=1,
            filter_num=36,
        )

        self.degconv_30 = EDTM(in_dim=64, nbins=8, cell_size=(8, 8))
        self.degconv_14 = EDTM(in_dim=64, nbins=8, cell_size=(4, 4))

        self.srf_module = MGFFM(
            embed_dims=[64, 64, 64, 64], mid_dim=16, size=(30, 30)
        )
        self.global_pool = torch.nn.AdaptiveAvgPool2d((1, 1))

        self.fully_connection_1 = torch.nn.Linear(7424, 2048)
        self.fully_connection_2 = torch.nn.Linear(2048, 1024)
        self.fully_connection_for_deg = torch.nn.Linear(64 * 2 + 1, 1024)

        self.dropout = torch.nn.Dropout(p=0.5)
        self.arcface = ArcMarginProduct(in_features=1024, out_features=label_num)

    def forward(self, feature_tensor, target=None):
        processed_feature, attention_weights = self.processing(feature_tensor)
        feature_tensor = self.dropout(processed_feature)
        feature_tensor = self.arcface(feature_tensor, target)
        return feature_tensor, F.normalize(feature_tensor, dim=-1), attention_weights

    def get_feature_vector(self, feature_tensor):
        feature_tensor, _ = self.processing(feature_tensor)
        return F.normalize(feature_tensor, p=2, dim=1, eps=1e-8)

    def processing(self, feature_tensor):
        local_feat_vector, first_order, second_order = self.feature_extraction(feature_tensor)

        enhanced_30 = self.degconv_30(first_order)
        enhanced_14 = self.degconv_14(second_order)

        srf_x2 = F.adaptive_avg_pool2d(enhanced_14, output_size=(21, 21))
        srf_x4 = F.adaptive_avg_pool2d(enhanced_14, output_size=(8, 8))
        srf_map = self.srf_module([enhanced_30, srf_x2, enhanced_14, srf_x4])

        gate_30 = torch.sigmoid(srf_map)
        gate_14 = F.interpolate(gate_30, size=enhanced_14.shape[-2:], mode='bilinear', align_corners=False)
        enhanced_30 = enhanced_30 * gate_30 + enhanced_30
        enhanced_14 = enhanced_14 * gate_14 + enhanced_14

        global_feat_30 = self.global_pool(enhanced_30).flatten(1)
        global_feat_14 = self.global_pool(enhanced_14).flatten(1)
        global_feat_srf = self.global_pool(srf_map).flatten(1)

        global_feat = torch.cat([global_feat_30, global_feat_14, global_feat_srf], dim=1)
        global_feat = self.fully_connection_for_deg(global_feat)

        local_feat = self.fully_connection_1(local_feat_vector)
        local_feat = self.fully_connection_2(local_feat)

        fused_feat = local_feat * self.weight + global_feat * (1 - self.weight)
        attention_weights = F.adaptive_avg_pool2d(gate_30, output_size=(12, 12)).flatten(1)

        return fused_feat, attention_weights
