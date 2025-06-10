import torch
import time
import numpy as np
import os
import open3d as o3d
import trimesh
from models import sampling
from models.utils import compute_circumsphere_centers
from models.meshprocessing import add_mid_vertices

def turn_manifold(mesh_path):
    mesh = trimesh.load_mesh(mesh_path)
    mesh = add_mid_vertices(mesh)
    mesh.export(mesh_path)

def read_delaunay_data(data_root):
    file_names = dict()
    file_names['cell_vertex_id'] = os.path.join(data_root, "cell_vertex_id.txt")
    file_names['cell_adj_id'] = os.path.join(data_root, "cell_adj_id.txt")
    file_names['infinite_cell_id'] = os.path.join(data_root, "infinite_cell_id.txt")
    file_names['infinite_vertex_position'] = os.path.join(data_root, "infinite_vertex.txt")

    cell_vertex_id = np.fromfile(file_names['cell_vertex_id'], dtype=int, sep=' ').reshape(-1, 4)
    cell_adj_id = np.fromfile(file_names['cell_adj_id'], dtype=int, sep=' ').reshape(-1, 4)
    infinite_cell_id = np.fromfile(file_names['infinite_cell_id'], dtype=int, sep=' ').reshape(-1)
    infinite_vertex_position = np.fromfile(file_names['infinite_vertex_position'], dtype=float, sep=' ').reshape(-1,3)

    cell_vertex_id = torch.tensor(cell_vertex_id).long().cpu()
    cell_adj_id = torch.tensor(cell_adj_id).long().cpu()
    infinite_cell_id = torch.tensor(infinite_cell_id).long().cpu()
    infinite_vertex_position = torch.tensor(infinite_vertex_position, dtype=torch.float).cpu()

    return cell_vertex_id, cell_adj_id, infinite_cell_id, infinite_vertex_position

def cal_queries(pcd_path, cell_vertex_id, infinite_cell_id, infinite_vertex_position, k_samples):
    pcd = o3d.io.read_point_cloud(pcd_path)
    points = torch.tensor(np.array(pcd.points), dtype=torch.float).cpu()

    cell_vertex = points[cell_vertex_id]  # n,4,3

    infinite_cell_vertex_id = cell_vertex_id[infinite_cell_id]  # m,4
    infinite_cell_vertex = points[infinite_cell_vertex_id]  # m,4,3
    indices = torch.where(infinite_cell_vertex_id == -1)
    infinite_cell_vertex[indices[0], indices[1], :] = infinite_vertex_position[indices[0], :]
    cell_vertex[infinite_cell_id] = infinite_cell_vertex  # n,4,3

    samples = sampling.random_sampling(cell_vertex.cpu(), k_samples).cpu()
    ball_centers = torch.mean(cell_vertex, dim=1, keepdim=True)  # n,1,3
    try:
        c_centers = compute_circumsphere_centers(cell_vertex).unsqueeze(1)  # n,1,3
        samples = torch.cat([samples, ball_centers, c_centers], dim=1)  # n,k+2,3
    except:
        c_centers = None
        samples = torch.cat([samples, ball_centers], dim=1)  # n,k+1,3

    return samples, c_centers

def constrant_labeling(centers_sdf, sdf_threshold):  # n,2,1
    ref = torch.where(centers_sdf >= sdf_threshold, 0.5, -0.5)
    ref = torch.sum(ref, dim=1)  # n,1; -1.,0.,1.

    return ref

def labeling(sdf_threshold, sdf_network, queries, c_centers, sdf_iter_step, ref_thresold):
    n,k,_ = queries.size()
    queries = queries.view(-1, 3).cuda()  # n*k,3
    max_segment_length = 10 * 10000
    segments = torch.split(queries, max_segment_length)
    sdf_list = []
    for segment in segments:
        sdf = sdf_network(segment, sdf_iter_step).detach()
        sdf_list.append(sdf)
        torch.cuda.empty_cache()
    sdf = torch.cat(sdf_list, dim=0).view(n,k,1).cpu()
    ref = torch.where(sdf[:,:-2,:] >= sdf_threshold, 1.0, 0.0)
    ref_sum = torch.mean(ref, dim=1)  # n,1
    pre_labels = torch.where(ref_sum >= ref_thresold, int(1), int(0))  # n,1
    if c_centers is None:
        labels = pre_labels
    else:
        con_labels = constrant_labeling(sdf[:,-2:,:], sdf_threshold)
        labels = torch.where(con_labels == 1., con_labels, pre_labels)
        labels = torch.where(con_labels == -1., 0, labels)

    return labels

def relabeling(labels, infinite_cell_id, cell_adj_id, conf):
    adj_labels = labels[cell_adj_id].squeeze(-1)  # n,4
    adj_labels_sum = torch.sum(adj_labels, dim=-1, keepdim=True)  # n,1
    inside = torch.where((adj_labels_sum == 0) | (adj_labels_sum == 1))
    outside = torch.where((adj_labels_sum == 3) | (adj_labels_sum == 4))
    labels[inside] = int(0)
    labels[outside] = int(1)
    inside_outside = conf.get_bool("model.sdf_network.inside_outside")
    labels[infinite_cell_id] = int(0) if inside_outside else int(1)

    return labels

def save_labels(labels, label_path):
    np.savetxt(label_path, labels.cpu().numpy(), fmt='%d')

def delaunay_meshing(data_dir, sdf_level, points, sdf_network, iter_step, k_samples, sdf_iter_step, conf):
    file_dir = os.path.join(data_dir, 'files')
    os.makedirs(file_dir, exist_ok=True)
    pcd = trimesh.PointCloud(vertices=points)
    point_path = os.path.join(file_dir, '{:0>8d}_vertices.ply'.format(iter_step))
    pcd.export(point_path)

    os.system("./models/delaunay_meshing/create_delaunay/create_delaunay %s %s" % (point_path, file_dir))
    cell_vertex_id, cell_adj_id, infinite_cell_id, infinite_vertex_position = read_delaunay_data(file_dir)

    queries, c_centers = cal_queries(point_path, cell_vertex_id, infinite_cell_id, infinite_vertex_position, k_samples)
    pre_labels = labeling(sdf_level, sdf_network, queries, c_centers, sdf_iter_step, ref_thresold=0.45)  # 0.45
    re_labels = relabeling(pre_labels, infinite_cell_id, cell_adj_id, conf)
    label_path = os.path.join(file_dir, "pre_label.txt")
    save_labels(re_labels, label_path)

    mesh_path = os.path.join(data_dir, '{:0>8d}_mesh_sdf{:.6f}.ply'.format(iter_step, sdf_level))
    os.system("./models/delaunay_meshing/create_mesh/create_mesh %s %s %s" % (file_dir, label_path, mesh_path))
    os.system("rm -rf %s" % file_dir)
    turn_manifold(mesh_path)

    return mesh_path
