import torch
import os
import json
from unet import UNet, ConvPass

setup_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

with open(os.path.join(setup_dir, 'config.json'), 'r') as f:
    config = json.load(f)


class AffsUNet(torch.nn.Module):

    def __init__(
            self,
            in_channels,
            num_fmaps,
            fmap_inc_factor,
            downsample_factors,
            kernel_size_down,
            kernel_size_up,
            padding="valid"):

        super().__init__()

        self.unet = UNet(
                in_channels=in_channels,
                num_fmaps=num_fmaps,
                fmap_inc_factor=fmap_inc_factor,
                downsample_factors=downsample_factors,
                kernel_size_down=kernel_size_down,
                kernel_size_up=kernel_size_up,
                constant_upsample=True,
                padding=padding)

        self.affs_head = ConvPass(num_fmaps, len(config['neighborhood']), [[1, 1, 1]], activation='Sigmoid')

    def forward(self, input):

        z = self.unet(input)
        affs = self.affs_head(z)

        return affs


class WeightedMSELoss(torch.nn.Module):

    def __init__(self):
        super(WeightedMSELoss, self).__init__()

    def forward(self, pred, target, weights):

        scale = (weights * (pred - target) ** 2)

        if len(torch.nonzero(scale)) != 0:

            mask = torch.masked_select(scale, torch.gt(weights, 0))
            loss = torch.mean(mask)

        else:

            loss = torch.mean(scale)

        return loss
