import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.utils import get_neighbor_idx_noself, get_neighbor_idx, PCA
from models.modules.netutils import gradient, guassian_kernel


def curve_growth_function(x, total_iter):
    y = (np.exp(x / total_iter) - 1) / (np.exp(1) - 1)

    return y

def pull_knn_loss(points, samples_moved, samples):  # n,k,3;n,3
    loss_pull = torch.linalg.norm((points - samples_moved.unsqueeze(1)), ord=2, dim=-1)  # n,k
    g_weight = guassian_kernel(points, samples).detach()
    loss_pull = (loss_pull * g_weight).sum(dim=-1).mean()

    return loss_pull

def grad_consis_knn_loss(gradient_points_norm, gradients_samples_norm, points, samples):  # n,k,3;n,3;n,k,3;n,3
    loss_consis = (1 - F.cosine_similarity(gradient_points_norm, gradients_samples_norm.unsqueeze(1), dim=-1))  # n,k
    g_weight = guassian_kernel(points, samples).detach()
    loss_consis = (loss_consis * g_weight).sum(dim=-1).mean()

    return loss_consis

def eikonal_loss(gradients_samples):
    loss_eikonal = ((gradients_samples.norm(2, dim=-1) - 1).square()).mean()

    return loss_eikonal

def digs_losses(loss_pull, loss_sdf, loss_inter, loss_eikonal, loss_div):
    w = [3e4, 3e3, 1e2, 5e1, 1e2]  #npull, sdf, intern, eikonal, div; digs loss
    loss = w[0]*loss_pull + w[1]*loss_sdf + w[2]*loss_inter + w[3]*loss_eikonal + w[4]*loss_div

    return loss

def div_loss(samples, gradients_samples):
    div_dx = gradient(samples, gradients_samples[:, 0])
    div_dy = gradient(samples, gradients_samples[:, 1])
    div_dz = gradient(samples, gradients_samples[:, 2])
    divergence = div_dx[:, 0] + div_dy[:, 1] + div_dz[:, 2]
    loss_div = (torch.clamp(torch.square(divergence), 0.1, 50)).mean()

    return loss_div

def sdf_loss(sample_sdf):
    loss = torch.abs(sample_sdf).mean()

    return loss

def cal_curvature_with_covariance(pts, knn):  # pts: n,3
    neigh_idx_noself = get_neighbor_idx_noself(pts.detach().cpu().numpy(), pts.detach().cpu().numpy(), knn) # n,k
    neigh_pts_noself = pts[neigh_idx_noself]  # n,k,3
    pts_dif = neigh_pts_noself - pts.unsqueeze(1)  # n,k,3
    pts_dif_T = pts_dif.permute(0, 2, 1)  # n,3,k
    co_matrix = pts_dif_T @ pts_dif  # n,3,3
    eigs, vectors = torch.linalg.eig(co_matrix)  # eigs:n,3;vectors:n,3,3
    eigs = eigs.real
    eigs_min_idx = torch.min(eigs, dim=1).indices  # n
    tensor_line = torch.arange(0, eigs.shape[0])
    eigs_min = eigs[tensor_line, eigs_min_idx]  # n
    pts_curvature = eigs_min / torch.sum(eigs, dim=1)
    pts_curvature /= 2. * pts_curvature.mean()

    # mean curvature
    neigh_pts = pts[neigh_idx_noself]  # n,k,3
    neigh_curvature = pts_curvature[neigh_idx_noself]  # n,k
    guassian_weight = guassian_kernel(neigh_pts, pts).detach()
    pts_curvature_ave = (neigh_curvature * guassian_weight).sum(dim=-1).unsqueeze(-1)  # n,1

    pts_curvature_ave = torch.sigmoid(pts_curvature_ave-torch.mean(pts_curvature_ave))
    pts_curvature_ave = (pts_curvature_ave - torch.min(pts_curvature_ave) + 1e-6) / (torch.max(pts_curvature_ave) - torch.min(pts_curvature_ave) + 1e-6)

    return pts_curvature_ave  # n,1

def cal_curvature_with_normal(pts, normals, knn):  # pts: n,3
    neigh_idx_noself = get_neighbor_idx_noself(pts.detach().cpu().numpy(), pts.detach().cpu().numpy(), knn) # n,k
    neigh_pts = pts[neigh_idx_noself]  # n,k,3
    neigh_normals = normals[neigh_idx_noself]  # n,k,3
    neigh_curvature = 1 - F.cosine_similarity(normals.unsqueeze(1), neigh_normals, dim=-1)  # n,k
    guassian_weight = guassian_kernel(neigh_pts, pts).detach()
    pts_curvature_ave = (neigh_curvature * guassian_weight).sum(dim=-1).unsqueeze(-1)  # n,1

    pts_curvature_ave = torch.sigmoid(pts_curvature_ave-torch.mean(pts_curvature_ave))
    pts_curvature_ave = (pts_curvature_ave - torch.min(pts_curvature_ave) + 1e-6) / (torch.max(pts_curvature_ave) - torch.min(pts_curvature_ave) + 1e-6)

    return pts_curvature_ave  # n,1

def cal_curvature_with_sdf(pts, eigenvalues, eigenvectors, knn):
    ei_values = eigenvalues.abs()    # n,3
    eigs_max_idx = torch.min(ei_values, dim=1).indices  # n
    tensor_line = torch.arange(0, ei_values.shape[0])
    eigs_max = ei_values[tensor_line, eigs_max_idx]  # n
    curvature = eigs_max / torch.sum(ei_values, dim=1)  # n
    # curvature = ei_values.mean(dim=-1)   # n
    curvature /= 2. * curvature.mean()
    ei_vectors = F.normalize(eigenvectors, dim=-1)

    # knn gaussian mean
    neigh_idx_noself = get_neighbor_idx_noself(pts.detach().cpu().numpy(), pts.detach().cpu().numpy(), knn)  # n,k
    neigh_pts = pts[neigh_idx_noself]  # n,k,3
    neigh_cur = curvature[neigh_idx_noself]  # n,k
    guassian_weight = guassian_kernel(neigh_pts, pts).detach()
    average_cur = (neigh_cur * guassian_weight).sum(dim=-1).unsqueeze(1)  # n,1

    return curvature, average_cur, ei_values, ei_vectors

def cal_cur_loss(curvature_surface, curvature_sample, sur_neigh_idx):
    sur_neigh_cur = curvature_sample[sur_neigh_idx]  # n  find nearest x2 for each x1
    cur_loss = torch.abs(curvature_surface - sur_neigh_cur) ** 2
    cur_loss = torch.sqrt(cur_loss.mean())

    return cur_loss

def cal_nc_loss_knn(surface_pts, surface_normals, sample_pts, sample_normals, knn):
    neigh_idx = get_neighbor_idx(sample_pts.detach().cpu().numpy(), surface_pts.detach().cpu().numpy(), knn)
    neigh_samples = sample_pts[neigh_idx]      # n,k,3
    neigh_normals = sample_normals[neigh_idx]  # n,k,3
    guassian_weight = guassian_kernel(neigh_samples, surface_pts).unsqueeze(-1).detach()   # n,k,1
    normal_weighted_ave = (neigh_normals * guassian_weight).sum(dim=1)  # n,3

    nc_loss = 1 - F.cosine_similarity(normal_weighted_ave, surface_normals, dim=-1).mean()

    return nc_loss   # 0.0001

def cal_nc_loss_with_PCA(surface_pts, surface_normals, sample_pts, conf):
    normals_pca = PCA(sample_pts, surface_pts, conf)
    product = (normals_pca * surface_normals).sum(-1)
    index = torch.where(product < 0)
    normals_pca[index] *= -1.
    nc_loss = 1 - F.cosine_similarity(normals_pca, surface_normals, dim=-1).mean()

    return nc_loss

def cal_nc_loss(surface_normals, sample_normals, sur_neigh_idx):
    neigh_normals = sample_normals[sur_neigh_idx]  # n,3
    nc_loss = 1 - F.cosine_similarity(surface_normals, neigh_normals, dim=-1).mean()

    return nc_loss

def cal_chamfer_loss(sur_pts, sample_pts, curvature_surface, loss_w, nearest_clamp):
    # find nearest x2 for each x1
    weight_sur_curvature = curvature_surface
    sur_neigh_idx = get_neighbor_idx(sample_pts.detach().cpu().numpy(), sur_pts.detach().cpu().numpy(), 1)  # n
    sur_neigh_pts = sample_pts[sur_neigh_idx]  # n,3
    dist_1 = torch.linalg.norm((sur_pts - sur_neigh_pts), ord=2, dim=-1).unsqueeze(-1) ** 2
    loss_part_1 = loss_w[0]*(dist_1 * weight_sur_curvature).mean() + loss_w[1]*dist_1.mean()

    # find nearest x1 for each x2
    sample_neigh_idx = get_neighbor_idx(sur_pts.detach().cpu().numpy(), sample_pts.detach().cpu().numpy(), 1)
    sample_neigh_pts = sur_pts[sample_neigh_idx]  # m,3
    dist_2 = torch.linalg.norm((sample_pts - sample_neigh_pts), ord=2, dim=-1) ** 2
    loss_part_2 = loss_w[2]*dist_2.mean()

    # find nearest x2 for each x2
    sample_neigh_idx_self = get_neighbor_idx_noself(sample_pts.detach().cpu().numpy(),sample_pts.detach().cpu().numpy(),1)
    sample_neigh_pts_self = sample_pts[sample_neigh_idx_self]
    relative_dist = sample_neigh_pts_self - sample_pts
    norm_dist = torch.linalg.norm(relative_dist, ord=2, dim=-1) ** 2
    norm_dist_ = torch.clamp(norm_dist, max=nearest_clamp*norm_dist.mean())
    loss_part_3 = -norm_dist_.mean() * loss_w[3]

    chamfer_loss = loss_part_1 + loss_part_2 + loss_part_3

    return chamfer_loss, sur_neigh_idx, sample_neigh_idx

def cal_vg_loss(sur_pts, sur_normals, curvature_ave, sample_pts, sample_grad, conf):
    loss_w = conf.get_list("train.loss_weight")
    nearest_clamp = conf.get_float("train.nearest_clamp")
    chamfer_loss, sur_neigh_idx, sample_neigh_idx = cal_chamfer_loss(sur_pts, sample_pts, curvature_ave, loss_w, nearest_clamp)
    sample_normal = F.normalize(sample_grad.detach(), dim=-1)
    normal_loss = cal_nc_loss(sur_normals, sample_normal, sur_neigh_idx)

    loss = chamfer_loss + loss_w[4]*normal_loss

    return loss