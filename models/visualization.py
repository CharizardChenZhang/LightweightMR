import numpy as np
import torch
import torch.nn.functional as F
import random
import open3d as o3d

def visible_points_curvature(pts, curvature, name):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pts.numpy())
    colors = curvature.numpy() * np.array([1., 0, 0])
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(name, point_cloud)