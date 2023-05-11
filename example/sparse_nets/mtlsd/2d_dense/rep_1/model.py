import torch
from unet import UNet, ConvPass


class MtlsdModel(torch.nn.Module):

    def __init__(
            self,
            in_channels,
            output_shapes,
            fmap_inc_factor,
            downsample_factors,
            kernel_size_down,
            kernel_size_up):

        super().__init__()

        num_fmaps = sum(output_shapes)

        if num_fmaps < 5:
            num_fmaps = 5

        self.unet = UNet(
                in_channels=in_channels,
                num_fmaps=num_fmaps,
                fmap_inc_factor=fmap_inc_factor,
                downsample_factors=downsample_factors,
                kernel_size_down=kernel_size_down,
                kernel_size_up=kernel_size_up,
                constant_upsample=True,
                num_heads=2)

        self.lsd_head = ConvPass(num_fmaps, output_shapes[0], [[1, 1]], activation='Sigmoid')
        self.aff_head = ConvPass(num_fmaps, output_shapes[1], [[1, 1]], activation='Sigmoid')

    def forward(self, input):

        z = self.unet(input)

        lsds = self.lsd_head(z[0])
        affs = self.aff_head(z[1])

        return lsds,affs


class WeightedMSELoss(torch.nn.Module):

    def __init__(self):
        super(WeightedMSELoss, self).__init__()

    def _calc_loss(self, pred, target, weights):

        scale = (weights * (pred - target) ** 2)

        if len(torch.nonzero(scale)) != 0:

            mask = torch.masked_select(scale, torch.gt(weights, 0))
            loss = torch.mean(mask)

        else:

            loss = torch.mean(scale)

        return loss

    def forward(
            self,
            lsds_prediction,
            lsds_target,
            lsds_weights,
            affs_prediction,
            affs_target,
            affs_weights):

        lsds_loss = self._calc_loss(lsds_prediction, lsds_target, lsds_weights)
        affs_loss = self._calc_loss(affs_prediction, affs_target, affs_weights)

        return lsds_loss + affs_loss
