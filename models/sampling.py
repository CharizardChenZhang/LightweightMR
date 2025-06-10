import torch
import numpy as np
import math
import torch.nn.functional as F
from models.utils import get_neighbor_idx_noself, get_neighbor_idx

# multi-label sampling
def random_sampling(cell_vertex, k_samples):
    random_size = k_samples
    random = torch.rand((random_size, 4)).cpu()  # k,4
    random = random / torch.sum(random, dim=1, keepdim=True)
    random_samples = cell_vertex.unsqueeze(1) * random.unsqueeze(-1).unsqueeze(0)  #n,k,4,3
    random_samples = torch.sum(random_samples, dim=2)  # n,k,3

    return random_samples

def max_lenghth_side_sampling(cell_vertex, centers):
    relative_dist = cell_vertex - centers  # n,4,3
    dist_norm = torch.linalg.norm(relative_dist, ord=2, dim=-1)  # n,4
    max_index = torch.argmax(dist_norm, dim=-1)  # n
    max_relative = torch.gather(relative_dist, 1, max_index.view(-1, 1, 1).expand(-1, 1, relative_dist.size(-1)))  # n,1,3
    samples = max_relative/2.0 + centers

    return samples

def interval_sampling(cell_vertex, centers, k_samples):
    offset_w = torch.linspace(0,1,k_samples+2)[1:-1]  # k
    samples = (cell_vertex - centers).unsqueeze(2) * offset_w.resize(1,1,k_samples,1) + centers.unsqueeze(2) # n,4,k,3
    samples = samples.reshape(-1, 4*k_samples, 3)  # n,4*k,3

    return samples

def mid_side_sampling(cell_vertex, centers):
    relative_dist = cell_vertex - centers  # n,4,3
    samples = relative_dist/2.0 + centers  # n,4,3

    return samples

# meshing strategy
def up_sampling(curvature, pts, point_gt, sample_size):
    curvature, pts, point_gt = curvature.detach(), pts.detach(), point_gt.detach()
    up_sampling_size = sample_size - len(pts)
    top_values, top_indices = torch.topk(curvature.view(-1), k=up_sampling_size, largest=True)
    curvature_best_points = pts[top_indices]
    idx = get_neighbor_idx(point_gt.cpu().numpy(), curvature_best_points.cpu().numpy(), 1)  # n
    up_sampling_points = point_gt[idx]
    sample_points = torch.cat((pts, up_sampling_points), dim=0)
    top_values = torch.cat((curvature, top_values.unsqueeze(-1)), dim=0)

    return sample_points, top_values

def up_sampling_2(sample_normals, samples, normals_gt, point_gt, sample_size):
    sample_normals, samples, normals_gt, point_gt = sample_normals.detach(), samples.detach(), normals_gt.detach(), point_gt.detach()
    sample_normals = F.normalize(sample_normals, dim=-1)
    sur_neigh_idx = get_neighbor_idx(samples.cpu().numpy(), point_gt.cpu().numpy(), 1)  # n
    neigh_normals = sample_normals[sur_neigh_idx]  # n,3
    normal_s = F.cosine_similarity(normals_gt, neigh_normals, dim=-1)
    up_sampling_size = sample_size - len(samples)
    candidate_size = (len(point_gt) // len(samples)) * up_sampling_size
    top_values, top_indices = torch.topk(normal_s.view(-1), k=candidate_size, largest=True)
    candidate_points = point_gt[top_indices]
    candidate_normals = normals_gt[top_indices]
    up_sampling_idx = farthest_point_sample(candidate_points.unsqueeze(0).cpu(), up_sampling_size).squeeze(0)
    up_sampling_points = candidate_points[up_sampling_idx]
    up_sampling_normals = candidate_normals[up_sampling_idx]
    sample_points = torch.cat((samples, up_sampling_points), dim=0)
    normals = torch.cat((sample_normals, up_sampling_normals), dim=0)

    return sample_points, normals

def up_sampling_3(sample_normals, samples, normals_gt, point_gt, sample_size):
    sample_normals, samples, normals_gt, point_gt = sample_normals.detach(), samples.detach(), normals_gt.detach(), point_gt.detach()
    sample_normals = F.normalize(sample_normals, dim=-1)
    sur_neigh_idx = get_neighbor_idx(samples.cpu().numpy(), point_gt.cpu().numpy(), 1)  # n
    neigh_normals = sample_normals[sur_neigh_idx]  # n,3
    normal_s = F.cosine_similarity(normals_gt, neigh_normals, dim=-1)
    _, indices = torch.sort(normal_s, descending=True)
    sort_neigh_idx = sur_neigh_idx[indices]
    unique_neigh_idx = torch.unique(sort_neigh_idx)
    up_sampling_size = sample_size - len(samples)
    candidate_points = (samples[unique_neigh_idx])[:up_sampling_size]
    up_sampling_idx = get_neighbor_idx(point_gt.cpu().numpy(), candidate_points.cpu().numpy(), 1)  # n
    up_sampling_points = point_gt[up_sampling_idx]
    up_sampling_normals = normals_gt[up_sampling_idx]
    sample_points = torch.cat((samples, up_sampling_points), dim=0)
    normals = torch.cat((sample_normals, up_sampling_normals), dim=0)

    return sample_points, normals

def down_sampling(curvature, pts, point_gt, sample_size):
    curvature, pts, point_gt = curvature.detach(), pts.detach(), point_gt.detach()
    top_values, top_indices = torch.topk(curvature.view(-1), k=int(sample_size*0.8), largest=True)
    sample_points = pts[top_indices]

    top_values_, top_indices_ = torch.topk(curvature.view(-1), k=sample_size-int(sample_size * 0.8), largest=True)
    sample_points_ = pts[top_indices_]
    idx = get_neighbor_idx(point_gt.cpu().numpy(), sample_points_.cpu().numpy(), 1)  #n
    sample_points_ = point_gt[idx]

    sample_points = torch.cat((sample_points, sample_points_), dim=0)
    top_values = torch.cat((top_values, top_values_), dim=0)
    return sample_points, top_values.unsqueeze(-1)

# common sampling

def farthest_point_sample(xyz, npoint):
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10  # initial dist
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)  # random initialization
    batch_indices = torch.arange(B, dtype=torch.long).to(device)  # 0~(B-1)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        dist = torch.linalg.norm((xyz - centroid), ord=2, dim=-1)  # b,n,c
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.argmax(distance, -1)

    return centroids