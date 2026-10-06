from typing import List

from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
from torch import nn

from monai.networks.nets import UNETR

class PlainUnetr(nn.Module):
    def __init__(self,
                 input_channels: int,
                 num_classes: int,
                 img_size,
                 feature_size: int = 16,
                 hidden_size: int = 768,
                 mlp_dim: int = 3072,
                 num_heads: int = 12,
                 proj_type: str = "conv",
                 norm_name = "instance",
                 conv_block: bool = True,
                 res_block: bool = True,
                 dropout_rate: float = 0.0,
                 spatial_dims: int = 3,
                 qkv_bias: bool = False,
                 save_attn: bool = True,
                 deep_supervision = False):
        super().__init__()
        self.unetr = UNETR(input_channels, num_classes, img_size, feature_size=feature_size, hidden_size=hidden_size, mlp_dim=mlp_dim, num_heads=num_heads, proj_type=proj_type, norm_name=norm_name, conv_block=conv_block, res_block=res_block, dropout_rate=dropout_rate, spatial_dims=spatial_dims, qkv_bias=qkv_bias, save_attn=save_attn)

    def forward(self, x):
        return self.unetr(x)

    def compute_conv_feature_map_size(self, input_size):
        return input_size

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)