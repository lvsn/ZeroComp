import torch
from torchvision.transforms import functional as TF
from torch.nn import functional as F
import numpy as np


class RandomGammaCorrection(torch.nn.Module):
    def __init__(self, gamma_range=(1.8, 2.4)):
        super().__init__()
        self.gamma_range = gamma_range

    def forward(self, img):
        # The input should be linear image before gamma correction
        gamma = torch.randn(1) * (self.gamma_range[1] - self.gamma_range[0]) + self.gamma_range[0]
        img = img ** (1.0 / gamma)
        return torch.clamp(img, 0.0, 1.0)


class RandomHDRReexpose(torch.nn.Module):
    def __init__(self, quantile_range=(0.85, 0.9), reexpose_range=(0.5, 2.0)):
        super().__init__()
        self.quantile_range = quantile_range
        self.reexpose_range = reexpose_range

    def forward(self, img):
        # The input should be HDR image
        quantile = torch.randn(1) * (self.quantile_range[1] - self.quantile_range[0]) + self.quantile_range[0]
        reexpose = torch.randn(1) * (self.reexpose_range[1] - self.reexpose_range[0]) + self.reexpose_range[0]
        value_quantile = torch.quantile(img, quantile)
        alpha = reexpose / (value_quantile + 1e-10)
        return torch.clamp(img * alpha, 0.0, 1.0)


def hdr_reexpose(img, percentile=90, max_mapping=0.8):
    """
    :param img:         HDR image
    :param percentile:
    :param max_mapping:
    :return:
    """
    r_percentile = max(np.percentile(img, percentile), 0)
    alpha = max_mapping / (r_percentile + 1e-10)
    return alpha * img, alpha


source_max_mapping = np.clip(0.6 * (2 ** np.random.normal(0, 0.5)), 0.1, 2)
