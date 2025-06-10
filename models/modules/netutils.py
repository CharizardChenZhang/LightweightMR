import torch
import numpy as np
import torch.nn.functional as F

def freeze(network):
    for name, parameter in network.named_parameters():
        parameter.requires_grad = False

    return network


def unfreeze(network):
    for name, parameter in network.named_parameters():
        parameter.requires_grad = True

    return network


def gradient(inputs, outputs):
    inputs.requires_grad_(True)
    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    points_grad = torch.autograd.grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]
    return points_grad

def cal_cur_hessian(sdf_network, pts, sdf_iter_step):
    grad_sdf, pred_sdf = sdf_network.gradient(pts, step=sdf_iter_step)
    hessian_matrix = torch.zeros(len(pts), 3, 3)
    for i in range(3):
        hessian_matrix[:, i, :] = gradient(pts, grad_sdf[..., i].unsqueeze(1))

    eigenvalues, eigenvectors = torch.linalg.eigh(hessian_matrix)

    return F.normalize(grad_sdf.detach(), dim=-1), eigenvalues.detach(), eigenvectors.detach()

def guassian_kernel(points, queries):  # input n,k,3/n,3
    pts_dist = torch.linalg.norm((points - queries.unsqueeze(1)), ord=2, dim=-1)  # n,k
    h = pts_dist.mean(dim=-1, keepdim=True)  # n,1
    dist_exp = torch.exp(-pts_dist ** 2 / h ** 2)  # n,k
    gaussian_weight = dist_exp / dist_exp.sum(dim=-1).unsqueeze(-1)  # n,k

    return gaussian_weight