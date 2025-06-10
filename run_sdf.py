# -*- coding: utf-8 -*-

import time
import torch
import torch.nn.functional as F
from tqdm import tqdm
from models.dataset import DatasetNP
from models.sdfnet import SDFNetwork
import argparse
from pyhocon import ConfigFactory
import os
from shutil import copyfile
import numpy as np
import trimesh
from models.utils import get_root_logger, print_log, setup_seed
import models.losses as losses
import math
import mcubes
import warnings

warnings.filterwarnings("ignore")

class Runner:
    def __init__(self, args, conf_path, mode='train', checkpoint_name=None):
        self.device = torch.device('cuda')

        # Configuration
        self.conf_path = conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        f.close()

        self.conf = ConfigFactory.parse_string(conf_text)
        self.base_exp_dir = os.path.join(args.expdir, args.dataname, args.subdatadir)
        os.makedirs(self.base_exp_dir, exist_ok=True)

        self.dataset_np = DatasetNP(args.datadir, args.dataname, self.conf)
        self.point_size = self.dataset_np.point_size
        self.dataset_knn = self.dataset_np.dataset_knn
        self.dataname = args.dataname
        self.iter_step = 0
        print("dataset knn is ", self.dataset_knn)

        # Training parameters
        self.maxiter = self.conf.get_int('train.maxiter')
        self.save_freq = self.conf.get_int('train.save_freq')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_freq = self.conf.get_int('train.val_freq')
        self.batch_size = self.conf.get_int('train.batch_size')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)

        self.mode = mode

        # Networks
        self.sdf_network = SDFNetwork(self.point_size, **self.conf['model.sdf_network']).to(self.device)
        self.optimizer = torch.optim.Adam(self.sdf_network.parameters(), lr=self.learning_rate)

        # Backup codes and configs for debug
        if self.mode[:5] == 'train':
            self.file_backup()

        if checkpoint_name:
            self.load_checkpoint(checkpoint_name)

    def train(self):
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = os.path.join(os.path.join(self.base_exp_dir), f'sdf_{timestamp}.log')
        logger = get_root_logger(log_file=log_file, name='outs')
        self.logger = logger
        batch_size = self.batch_size

        loss_w = self.conf.get_list("train.loss_weight")
        model_type = self.conf.get("dataset.type")
        res_step = self.maxiter - self.iter_step
        for iter_i in tqdm(range(res_step)):
            self.update_learning_rate_np(iter_i)

            sample_near, points_near, normals_near, sample_uniform, points_uniform, normals_uniform, point_gt = self.dataset_np.sdf_train_data(batch_size, self.iter_step, model_type)
            # queries
            samples = torch.cat((sample_near, sample_uniform), dim=0)
            gradients_samples, sdf_samples = self.sdf_network.gradient(samples, self.iter_step)
            gradients_samples_norm = F.normalize(gradients_samples, dim=-1)
            samples_moved = samples - gradients_samples_norm * sdf_samples

            # Gradient Consistency loss
            move_position = samples_moved.detach()
            gradients_samples_moved, _ = self.sdf_network.gradient(move_position, self.iter_step)
            gradients_samples_moved_norm = F.normalize(gradients_samples_moved, dim=-1)
            loss_grad_consis = (1 - F.cosine_similarity(gradients_samples_moved_norm, gradients_samples_norm, dim=-1)).mean()

            # points
            points = torch.cat((points_near, points_uniform), dim=0)
            points_ = points.clone() if self.dataset_knn == 1 else points[:,0,:]
            sdf_points = self.sdf_network.sdf(points_, self.iter_step)

            # loss
            if self.dataset_knn == 1:
                loss_pull = torch.linalg.norm((points - samples_moved), ord=2, dim=-1).mean()
            else:
                loss_pull = losses.pull_knn_loss(points, samples_moved, samples)
            loss_sdf = torch.abs(sdf_points).mean()
            loss_inter = torch.exp(-1e2 * torch.abs(sdf_samples)).mean()
            if normals_near is not None and normals_uniform is not None:
                normals_gt = torch.cat((normals_near, normals_uniform), dim=0)
                loss_normal = (1 - F.cosine_similarity(normals_gt, gradients_samples, dim=-1)).mean()
            else:
                loss_normal = torch.zeros((1,), device=loss_sdf.device)

            # total loss
            loss = loss_w[0]*loss_pull + loss_w[1]*loss_sdf + loss_w[2]*loss_grad_consis + loss_w[3]*loss_inter + 0.01*loss_normal
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.iter_step += 1
            if self.iter_step % self.report_freq == 0:
                print_log('iter:{:8>d} loss = {} lr={}'.format(self.iter_step, loss, self.optimizer.param_groups[0]['lr']), logger=logger)

            if self.iter_step % self.val_freq == 0 and self.iter_step != 0:
                self.validate_mesh(resolution=512, threshold=0.0, real_world=True)

            if self.iter_step % self.save_freq == 0 and self.iter_step != 0:
                self.save_checkpoint()

    def validate_gradient(self, real_world=True, sdf_level=0.):
        N = 500_000
        queries = self.dataset_np.sample.split(N)
        queries_moved_list = []
        for batch in queries:
            for i in range(10):
                gradients_queries, sdf_queries = self.sdf_network.gradient(batch, self.iter_step)
                gradients_queries, sdf_queries = gradients_queries.detach(), sdf_queries.detach()
                gradients_queries_norm = F.normalize(gradients_queries, dim=-1)
                queries_moved = batch - gradients_queries_norm * (sdf_queries-sdf_level)
                batch = queries_moved.detach()
                torch.cuda.empty_cache()
            queries_moved_list.append(batch.clone())
        queries_moved_list = torch.cat(queries_moved_list, dim=0)

        points = queries_moved_list.detach().cpu().numpy()
        pcd = trimesh.PointCloud(vertices=points)
        if real_world:
            pcd.apply_scale(self.dataset_np.scale)
            pcd.apply_translation(self.dataset_np.loc)

        os.makedirs(os.path.join(self.base_exp_dir, 'sdf_valgrad'), exist_ok=True)
        pcd.export(os.path.join(self.base_exp_dir, 'sdf_valgrad', '{:0>8d}_{:.6f}.ply'.format(self.iter_step, sdf_level)))

    def validate_mesh(self, resolution=512, threshold=0.0, real_world=True):
        bound_min = torch.tensor(self.dataset_np.object_bbox_min, dtype=torch.float32)
        bound_max = torch.tensor(self.dataset_np.object_bbox_max, dtype=torch.float32)
        os.makedirs(os.path.join(self.base_exp_dir, 'sdf_outputs'), exist_ok=True)
        mesh = self.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold,
                                     query_func=lambda pts: -self.sdf_network.sdf(pts, self.iter_step))
        if real_world:
            mesh.apply_scale(self.dataset_np.scale)
            mesh.apply_translation(self.dataset_np.loc)

        mesh.export(os.path.join(self.base_exp_dir, 'sdf_outputs', '{:0>8d}_{}_res{}.ply'.format(self.iter_step, str(threshold), resolution)))

    def update_learning_rate_np(self, iter_step):
        warn_up = self.warm_up_end
        max_iter = self.maxiter
        init_lr = self.learning_rate
        lr = (iter_step / warn_up) if iter_step < warn_up else 0.5 * (math.cos((iter_step - warn_up) / (max_iter - warn_up) * math.pi) + 1)
        lr = lr * init_lr
        for g in self.optimizer.param_groups:
            g['lr'] = lr

    def extract_fields(self, bound_min, bound_max, resolution, query_func):
        N = 128
        X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
        Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
        Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

        u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
        with torch.no_grad():
            for xi, xs in enumerate(X):
                for yi, ys in enumerate(Y):
                    for zi, zs in enumerate(Z):
                        xx, yy, zz = torch.meshgrid(xs, ys, zs)
                        pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                        val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                        u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
        return u

    def extract_geometry(self, bound_min, bound_max, resolution, threshold, query_func):
        print('Creating mesh with threshold: {}'.format(threshold))
        u = self.extract_fields(bound_min, bound_max, resolution, query_func)
        vertices, triangles = mcubes.marching_cubes(u, threshold)
        b_max_np = bound_max.detach().cpu().numpy()
        b_min_np = bound_min.detach().cpu().numpy()

        vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
        mesh = trimesh.Trimesh(vertices, triangles)

        return mesh

    def file_backup(self):
        dir_lis = self.conf['general.recording']
        os.makedirs(os.path.join(self.base_exp_dir, 'recording'), exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.base_exp_dir, 'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == '.py':
                    copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))

        copyfile(self.conf_path, os.path.join(self.base_exp_dir, 'recording', 'sdf_config.conf'))

    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'sdf_checkpoints', checkpoint_name), map_location=self.device)
        print(os.path.join(self.base_exp_dir, 'sdf_checkpoints', checkpoint_name))
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.iter_step = checkpoint['iter_step']

    def save_checkpoint(self):
        checkpoint = {
            'sdf_network_fine': self.sdf_network.state_dict(),
            'iter_step': self.iter_step,
        }
        os.makedirs(os.path.join(self.base_exp_dir, 'sdf_checkpoints'), exist_ok=True)
        torch.save(checkpoint, os.path.join(self.base_exp_dir, 'sdf_checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/sdf.conf')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--subdatadir', type=str, default='SDF')
    parser.add_argument('--datadir', type=str, default='./example/data/')
    parser.add_argument('--expdir', type=str, default='./example/exp/')
    parser.add_argument('--dataname', type=str, default='47984')
    parser.add_argument('--checkpoint_name', type=str, default=None)
    args = parser.parse_args()

    setup_seed(123456)
    torch.cuda.set_device(args.gpu)
    runner = Runner(args, args.conf, args.mode, args.checkpoint_name)

    if args.mode == 'train':
        runner.train()
    elif args.mode == 'validate_mesh':
        threshs = [-0.001, 0.0]
        for thresh in threshs:
            runner.validate_mesh(resolution=32, threshold=thresh, real_world=True)
    elif args.mode == 'validate_gradient':
        runner.validate_gradient(real_world=True, sdf_level=0.0)