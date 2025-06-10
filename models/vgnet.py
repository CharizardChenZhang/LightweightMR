import torch
import torch.nn as nn
import numpy as np
from models.modules.embedder import get_embedder
from models.modules.triplane import Hash_triplane, Hash_grid
from models.modules.PointTransformerv3 import PointTransformerV3

class VGNetwork(nn.Module):
    def __init__(self,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 skip_in=(4,),
                 multires=0,
                 scale=1,
                 geometric_init=True,
                 weight_norm=True,
                 use_grid_feature=False,
                 use_plane_feature=False):
        super(VGNetwork, self).__init__()
        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embed_fn_fine = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn_fine = embed_fn
            dims[0] = input_ch + 3
        else:
            dims[0] += 3

        self.use_grid_feature = use_grid_feature
        if self.use_grid_feature:
            self.grid_encoding = Hash_grid(point_size=0, use_pro=False, max_levels=16)

        self.use_plane_feature = use_plane_feature
        if self.use_plane_feature:
            self.plane_encoding = Hash_triplane(point_size=0, multires=multires, use_pro=False, max_levels=16)

        if self.use_grid_feature or self.use_plane_feature:
            dims[0] += 16 * 2

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
            if weight_norm:
                lin = nn.utils.weight_norm(lin)
            setattr(self, "lin" + str(l), lin)

        self.activation = nn.ReLU()

    def forward(self, samples, normals, step):
        inputs = samples * self.scale
        if self.embed_fn_fine is not None:
            inputs = self.embed_fn_fine(inputs)
        inputs = torch.cat((inputs, normals), dim=-1)

        feature = 0.
        if self.use_plane_feature:
            feature += self.plane_encoding(inputs[..., :3], step)
        if self.use_grid_feature:
            feature += self.grid_encoding(inputs[..., :3], step)
        if self.use_plane_feature or self.use_grid_feature:
            inputs = torch.cat((inputs, feature), dim=-1)

        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)

            x = lin(x)
            if l < self.num_layers - 2:
                x = self.activation(x)

        moving_pcd = samples + x / self.scale

        return moving_pcd


class VGNetwork_PTF(nn.Module):
    def __init__(self,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 skip_in=(4,),
                 multires=0,
                 scale=1,
                 geometric_init=True,
                 weight_norm=True,
                 use_grid_feature=False,
                 use_plane_feature=False):
        super(VGNetwork_PTF, self).__init__()

        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embed_fn_fine = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn_fine = embed_fn

        self.use_grid_feature = use_grid_feature
        if self.use_grid_feature:
            self.grid_encoding = Hash_grid(point_size=0, use_pro=False, max_levels=16)

        self.use_plane_feature = use_plane_feature
        if self.use_plane_feature:
            self.plane_encoding = Hash_triplane(point_size=0, multires=multires, use_pro=False, max_levels=16)

        if self.use_grid_feature or self.use_plane_feature:
            dims[0] += 16 * 2

        self.transformer = PointTransformerV3(in_channels=input_ch+3)
        dims[0] += 64

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
            if weight_norm:
                lin = nn.utils.weight_norm(lin)
            setattr(self, "lin" + str(l), lin)

        self.activation = nn.ReLU()

    def forward(self, samples, normals, step):
        inputs = samples * self.scale
        if self.embed_fn_fine is not None:
            inputs = self.embed_fn_fine(inputs)
        inputs = torch.cat((inputs, normals), dim=-1)

        # point transformer v3
        point_dict = dict()
        offset = torch.tensor([len(inputs)]).cuda()
        point_dict['coord'] = inputs[..., :3].clone()
        point_dict['grid_size'] = 0.01
        point_dict['offset'] = offset
        point_dict['feat'] = inputs.clone()
        pts_feat_trans = self.transformer(point_dict)
        inputs = torch.cat((inputs[..., :3], pts_feat_trans['feat']), dim=-1)

        feature = 0.
        if self.use_plane_feature:
            feature += self.plane_encoding(inputs[..., :3], step)
        if self.use_grid_feature:
            feature += self.grid_encoding(inputs[..., :3], step)
        if self.use_plane_feature or self.use_grid_feature:
            inputs = torch.cat((inputs, feature), dim=-1)

        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)

            x = lin(x)
            if l < self.num_layers - 2:
                x = self.activation(x)

        moving_pcd = samples + x / self.scale

        return moving_pcd


