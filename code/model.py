import torch.nn as nn
import code.blocks as block


class EfRLFN(nn.Module):
    """
    Efifcient Residual Local Feature Network (EfRLFN)
    """

    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 feature_channels=52,
                 upscale=4):
        super(EfRLFN, self).__init__()

        self.conv_1 = block.conv_layer(in_channels,
                                       feature_channels,
                                       kernel_size=3)

        self.block_1 = block.ERLFB(feature_channels)
        self.block_2 = block.ERLFB(feature_channels)
        self.block_3 = block.ERLFB(feature_channels)
        self.block_4 = block.ERLFB(feature_channels)
        self.block_5 = block.ERLFB(feature_channels)
        self.block_6 = block.ERLFB(feature_channels)

        self.conv_2 = block.conv_layer(feature_channels,
                                       feature_channels,
                                       kernel_size=3)

        self.upsampler = block.pixelshuffle_block(feature_channels,
                                                  out_channels,
                                                  upscale_factor=upscale)

    def forward(self, x):
        out_feature = self.conv_1(x)

        out_b1 = self.block_1(out_feature)
        out_b2 = self.block_2(out_b1)
        out_b3 = self.block_3(out_b2)
        out_b4 = self.block_4(out_b3)
        out_b5 = self.block_5(out_b4)
        out_b6 = self.block_6(out_b5)

        out_low_resolution = self.conv_2(out_b6 + out_feature)
        output = self.upsampler(out_low_resolution)

        return output