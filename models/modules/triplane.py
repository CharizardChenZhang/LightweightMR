import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.modules.embedder import get_embedder
from models.hashencoder.hashgrid import HashEncoder


def set_level_auto(point_size):
    intervals = [0, 20_000, 100_000, 500_000]
    values = [6, 8, 10, 12]

    for i in range(len(intervals) - 1):
        if intervals[i] <= point_size < intervals[i + 1]:
            return values[i]
    return values[-1]

class Hash_triplane(nn.Module):
    def __init__(self, point_size, multires, divide_factor=1.0, use_pro=True, max_levels=None):
        super(Hash_triplane, self).__init__()
        encoding_2d_config = {'input_dim':2, 'num_levels':16, 'level_dim':2, 'per_level_scale':2,
                              'base_resolution':16, 'log2_hashmap_size':19, 'desired_resolution':2048}

        self.xy = HashEncoder(**encoding_2d_config)
        self.yz = HashEncoder(**encoding_2d_config)
        self.xz = HashEncoder(**encoding_2d_config)
        self.feat_dim = 16 * 2
        self.divide_factor = divide_factor
        self.use_pro = use_pro
        self.max_levels = set_level_auto(point_size) if max_levels is None else max_levels

        self.input_dim = self.feat_dim + 1
        if multires > 0:
            embed_fn, embed_dim = get_embedder(multires, input_dims=2)
            self.embed_fn_fine = embed_fn
            self.input_dim = self.feat_dim + embed_dim

        self.lin = nn.Linear(self.input_dim, self.feat_dim)
        torch.nn.init.constant_(self.lin.bias, 0.0)
        torch.nn.init.constant_(self.lin.weight[:, 3:], 0.0)
        torch.nn.init.normal_(self.lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(self.feat_dim))
        self.activation = nn.Softplus()

    def forward(self, input, step):
        mask = torch.zeros((1, self.feat_dim))
        level = min((step // 1000) + 1, self.max_levels)
        mask[:, 0:level * 2] = 1.0

        # require point range in [-1, 1], original point in [-0.5,0.5]
        input = input / self.divide_factor
        xy_feat, yz_feat, xz_feat = self.xy(input[:, [0, 1]]).clone(), self.yz(input[:, [1, 2]]).clone(), self.xz(input[:, [0, 2]]).clone()
        if self.use_pro:
            xy_feat, yz_feat, xz_feat = xy_feat*mask, yz_feat*mask, xz_feat*mask

        feature = xy_feat + yz_feat + xz_feat

        return feature

    def linear_embedding(self, input, xy_feat, yz_feat, xz_feat):
        xy_embed, yz_embed, xz_embed = input[:, [0, 1]], input[:, [1, 2]], input[:, [0, 2]]
        if self.embed_fn_fine is not None:
            xy_embed, yz_embed, xz_embed = self.embed_fn_fine(xy_embed), self.embed_fn_fine(yz_embed), self.embed_fn_fine(xz_embed)
        xy_feat, yz_feat, xz_feat = (torch.cat((xy_embed,xy_feat), dim=-1), torch.cat((yz_embed,yz_feat), dim=-1),
                                     torch.cat((xz_embed,xz_feat), dim=-1))
        feature = torch.cat((xy_feat.unsqueeze(1), yz_feat.unsqueeze(1), xz_feat.unsqueeze(1)), dim=1) # n,3,c
        feature = self.activation(self.lin(feature))
        feature = feature.reshape(len(input), -1)

        return feature

class Hash_grid(nn.Module):
    def __init__(self, point_size, divide_factor=1.0, use_pro=True, max_levels=None):
        super(Hash_grid, self).__init__()
        encoding_3d_config = {'input_dim':3, 'num_levels':16, 'level_dim':2, 'per_level_scale':2,
                              'base_resolution':16, 'log2_hashmap_size':19, 'desired_resolution':2048}

        self.hash_grid = HashEncoder(**encoding_3d_config)
        self.feat_dim = 16 * 2
        self.divide_factor = divide_factor
        self.use_pro = use_pro
        self.max_levels = set_level_auto(point_size) if max_levels is None else max_levels

    def forward(self, x, step):
        mask = torch.zeros((1, self.feat_dim))
        level = min((step // 1000) + 1, self.max_levels)
        mask[:, 0:level * 2] = 1.0

        # require point range in [-1, 1], original point in [-0.5,0.5]
        x = x / self.divide_factor
        feature = self.hash_grid(x).clone()
        if self.use_pro:
            feature *= mask

        return feature