import open3d as o3d
import torch
import torch.nn.functional as F
import numpy as np
import os
from models.cpplib.libkdtree import KDTree
import trimesh
from models import utils
import fpsample

def search_nearest_point(point_batch, point_gt):
    num_point_batch, num_point_gt = point_batch.shape[0], point_gt.shape[0]
    point_batch = point_batch.unsqueeze(1).repeat(1, num_point_gt, 1)
    point_gt = point_gt.unsqueeze(0).repeat(num_point_batch, 1, 1)

    distances = torch.sqrt(torch.sum((point_batch-point_gt) ** 2, axis=-1) + 1e-12) 
    dis_idx = torch.argmin(distances, axis=1).detach().cpu().numpy()

    return dis_idx

def process_data(data_dir, dataname, conf, with_normal=False):
    if os.path.exists(os.path.join(data_dir, dataname) + '.ply'):
        if with_normal:
            pcd = o3d.io.read_point_cloud(os.path.join(data_dir, dataname) + '.ply')
            pointcloud = np.array(pcd.points)
            pointnormal = np.array(pcd.normals)
            pointnormal = pointnormal / np.linalg.norm(pointnormal, axis=-1, keepdims=True)
        else:
            pointcloud = trimesh.load(os.path.join(data_dir, dataname) + '.ply').vertices
            pointcloud = np.asarray(pointcloud)
    elif os.path.exists(os.path.join(data_dir, dataname) + '.xyz'):
        pointcloud = np.load(os.path.join(data_dir, dataname)) + '.xyz'
    else:
        print('Only support .xyz or .ply data. Please make adjust your data.')
        exit()
    shape_scale = np.max(
        [np.max(pointcloud[:, 0]) - np.min(pointcloud[:, 0]), np.max(pointcloud[:, 1]) - np.min(pointcloud[:, 1]),
         np.max(pointcloud[:, 2]) - np.min(pointcloud[:, 2])])
    shape_center = [(np.max(pointcloud[:, 0]) + np.min(pointcloud[:, 0])) / 2,
                    (np.max(pointcloud[:, 1]) + np.min(pointcloud[:, 1])) / 2,
                    (np.max(pointcloud[:, 2]) + np.min(pointcloud[:, 2])) / 2]
    pointcloud = pointcloud - shape_center
    pointcloud = pointcloud / shape_scale

    queries_size = conf.get_int('dataset.queries_size')
    POINT_NUM = pointcloud.shape[0] // 60
    POINT_NUM_GT = pointcloud.shape[0] // 60 * 60
    QUERY_EACH = queries_size // POINT_NUM_GT

    point_idx = np.random.choice(pointcloud.shape[0], POINT_NUM_GT, replace=False)
    pointcloud = pointcloud[point_idx, :]
    if with_normal:
        pointnormal = pointnormal[point_idx, :]

    ptree = KDTree(pointcloud)
    sigmas = []
    for p in np.array_split(pointcloud, 100, axis=0):
        d = ptree.query(p, 51)
        sigmas.append(d[0][:, -1])

    sigmas = np.concatenate(sigmas)
    sample = []
    sample_near = []
    sample_near_normal = []
    kdtree = KDTree(pointcloud)
    knn = conf.get_int('dataset.pull_knn')
    for i in range(QUERY_EACH):
        scale = 0.25 * np.sqrt(POINT_NUM_GT / 20000)
        tt = pointcloud + scale * np.expand_dims(sigmas, -1) * np.random.normal(0.0, 1.0, size=pointcloud.shape)
        sample.append(tt)
        tt = tt.reshape(-1, 3)

        _, nearest_idx = kdtree.query(tt, knn)
        nearest_points = pointcloud[nearest_idx]
        nearest_points = np.asarray(nearest_points).reshape(-1, 3)
        sample_near.append(nearest_points)
        if with_normal:
            nearest_points_normals = np.asarray(pointnormal[nearest_idx]).reshape(-1, 3)
            sample_near_normal.append(nearest_points_normals)

    sample = np.asarray(sample).reshape(-1, 3)
    sample_near = np.asarray(sample_near).reshape(-1, 3)
    cube_boxsize = 1.1
    sample_uniform = np.random.rand(sample.shape[0] // 10, 3)
    sample_uniform = cube_boxsize * (sample_uniform - 0.5)  # [-0.55,0.55]
    _, nearest_idx = kdtree.query(sample_uniform, knn)
    nearest_points = pointcloud[nearest_idx]
    nearest_points = np.asarray(nearest_points).reshape(-1, 3)
    sample_uniform_near = nearest_points

    if with_normal:
        sample_near_normal = np.asarray(sample_near_normal).reshape(-1, 3)
        sample_uniform_near_normal = np.asarray(pointnormal[nearest_idx]).reshape(-1, 3)
    else:
        sample_near_normal = None
        sample_uniform_near_normal = None

    np.savez(os.path.join(data_dir, dataname) + '.npz', sample=sample, loc=shape_center, scale=shape_scale,
             sample_near=sample_near, sample_near_normal=sample_near_normal,
             sample_uniform=sample_uniform, sample_uniform_near=sample_uniform_near, sample_uniform_near_normal=sample_uniform_near_normal,
             point=pointcloud, knn=knn)

class DatasetNP:
    def __init__(self, datadir, dataname, conf):
        super(DatasetNP, self).__init__()
        self.device = torch.device('cuda')

        self.data_dir = datadir
        self.np_data_name = dataname + '.npz'

        self.with_normal = False  # if using normal, set to True
        if os.path.exists(os.path.join(self.data_dir, self.np_data_name)):
            print('Data existing. Loading data...')
        else:
            print('Data not found. Processing data...')
            process_data(self.data_dir, dataname, conf, with_normal=self.with_normal)
        load_data = np.load(os.path.join(self.data_dir, self.np_data_name))

        self.dataset_knn = load_data['knn']
        self.point = np.asarray(load_data['sample_near']).reshape(-1,self.dataset_knn,3).squeeze()
        self.sample = np.asarray(load_data['sample']).reshape(-1,3)
        self.point_gt = np.asarray(load_data['point']).reshape(-1,3)
        self.point_uniform = np.asarray(load_data['sample_uniform_near']).reshape(-1,self.dataset_knn,3).squeeze()
        self.sample_uniform = np.asarray(load_data['sample_uniform']).reshape(-1,3)
        if self.with_normal:
            self.point_normal = np.asarray(load_data['sample_near_normal']).reshape(-1,self.dataset_knn,3).squeeze()
            self.point_uniform_normal = np.asarray(load_data['sample_uniform_near_normal']).reshape(-1,self.dataset_knn,3).squeeze()
        self.loc = load_data['loc']
        self.scale = load_data['scale']

        self.point_size = len(self.point_gt)
        self.sample_points_num = self.sample.shape[0] - 1
        self.object_bbox_min = np.array([np.min(self.point_gt[:,0]), np.min(self.point_gt[:,1]), np.min(self.point_gt[:,2])]) -0.05
        self.object_bbox_max = np.array([np.max(self.point_gt[:,0]), np.max(self.point_gt[:,1]), np.max(self.point_gt[:,2])]) +0.05
        print('Data bounding box:',self.object_bbox_min,self.object_bbox_max)
    
        self.point = torch.from_numpy(self.point).to(self.device).float()
        self.sample = torch.from_numpy(self.sample).to(self.device).float()
        self.point_gt = torch.from_numpy(self.point_gt).to(self.device).float()
        self.point_uniform = torch.from_numpy(self.point_uniform).to(self.device).float()
        self.sample_uniform = torch.from_numpy(self.sample_uniform).to(self.device).float()
        if self.with_normal:
            self.point_normal = torch.from_numpy(self.point_normal).to(self.device).float()
            self.point_uniform_normal = torch.from_numpy(self.point_uniform_normal).to(self.device).float()

        print('NP Load data: End')

    def sdf_train_data(self, batch_size, iter_step, model_type):
        ## manner 1: uniform
        interval = (self.sample_points_num+1) // 100_000
        index_coarse = np.random.choice(interval, 1)
        index_fine = np.random.choice(self.sample_points_num//interval, batch_size, replace=False)
        near_index = index_fine * interval + index_coarse
        ## manner 2: random
        # near_index = np.random.choice(len(self.point), batch_size, replace=False)

        sample_near = self.sample[near_index]
        points_near = self.point[near_index]
        normals_near = None

        if model_type == "object":
            if iter_step < 2000:
                uni_interval = 1
            elif iter_step < 5000:
                uni_interval = 2
            elif iter_step < 8000:
                uni_interval = 4
            else:
                uni_interval = 8
        elif model_type == "scene":
            uni_interval = 10
        else:
            raise ValueError("model_type is wrong")

        uniform_index = np.random.choice(self.sample_uniform.shape[0]-1, batch_size // uni_interval, replace=False)
        sample_uniform = self.sample_uniform[uniform_index]
        points_uniform = self.point_uniform[uniform_index]
        normals_uniform = None

        if self.with_normal:
            normals_near = self.point_normal[near_index]
            normals_uniform = self.point_uniform_normal[uniform_index]
        return sample_near, points_near, normals_near, sample_uniform, points_uniform, normals_uniform, self.point_gt

    def get_surface_queries(self, conf, sdf_network, sdf_iter_step):
        queries_size = conf.get_int('dataset.surface_queries')
        sdf_level = conf.get_float('dataset.project_sdf_level')
        noisy_pts = conf.get_bool('dataset.noisy_pts')

        gt_size = len(self.point_gt)
        queries = self.point_gt.clone()
        if noisy_pts or sdf_level != 0.:
            queries = self.project_queries(queries, sdf_network, sdf_iter_step, sdf_level=sdf_level)

        if queries_size > gt_size:
            pad_size = queries_size - gt_size
            ## manner 1: uniform
            # pad_interval = max(self.sample_points_num // pad_size, 1)
            # pad_queries = self.sample[::pad_interval, :]
            # pad_queries = self.project_queries(pad_queries, sdf_network, sdf_iter_step, sdf_level=sdf_level)
            ## manner 2: fps
            pad_queries = self.sample.clone()
            pad_queries = self.project_queries(pad_queries, sdf_network, sdf_iter_step, sdf_level=sdf_level)
            pad_queries = self.fps_select_vertices(pad_queries, pad_size)

            queries = torch.cat((queries, pad_queries), dim=0)

        return queries.detach()

    def project_queries(self, queries, sdf_network, sdf_iter_step, sdf_level):
        N = 100_000
        queries = queries.split(N)
        queries_moved_list = []
        for batch in queries:
            for i in range(10):
                gradients_queries, sdf_queries = sdf_network.gradient(batch, sdf_iter_step)
                gradients_queries, sdf_queries = gradients_queries.detach(), sdf_queries.detach()
                gradients_queries_norm = F.normalize(gradients_queries, dim=-1)
                queries_moved = batch - gradients_queries_norm * (sdf_queries - sdf_level)
                batch = queries_moved.detach()
                torch.cuda.empty_cache()
            queries_moved_list.append(batch.clone())
        surface_reference = torch.cat(queries_moved_list, dim=0)

        return surface_reference

    def fps_select_vertices(self, point_gt, batch_size):
        sample_idx = torch.tensor(fpsample.bucket_fps_kdtree_sampling(point_gt.detach().cpu().numpy(), batch_size).astype(float)).cuda()
        sample_points = point_gt[sample_idx.long()].detach()

        return sample_points

    def uniform_select_vertices(self, point_gt, batch_size):
        interval = len(point_gt) // batch_size
        sample_points = point_gt[::interval, :].detach()

        return sample_points

    def cal_nearest_clamp(self, sample_pts):
        sample_neigh_idx_self = utils.get_neighbor_idx_noself(sample_pts.detach().cpu().numpy(), sample_pts.detach().cpu().numpy(), 1)
        sample_neigh_pts_self = sample_pts[sample_neigh_idx_self]
        relative_dist = sample_neigh_pts_self - sample_pts
        norm_dist = torch.linalg.norm(relative_dist, ord=2, dim=-1) ** 2
        nearest_clamp = norm_dist.mean().detach()

        return nearest_clamp



