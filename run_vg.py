# -*- coding: utf-8 -*-

import os
import time
import torch
import torch.nn.functional as F
import argparse
from pyhocon import ConfigFactory
import numpy as np
import trimesh
from tqdm import tqdm
from models.dataset import DatasetNP
from models.sdfnet import SDFNetwork
from models.vgnet import VGNetwork, VGNetwork_PTF
from models.utils import get_root_logger, print_log, setup_seed
from models.meshing import delaunay_meshing
from models.modules import netutils
from models import visualization, sampling, losses
import math
import warnings
from shutil import copyfile
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
        self.sdf_exp_dir = os.path.join(args.expdir, args.dataname, args.sdf_subdatadir)
        self.base_exp_dir = os.path.join(args.expdir, args.dataname, args.subdatadir)
        if not os.path.exists(self.sdf_exp_dir):
            raise ValueError("path does not exist")
        os.makedirs(self.base_exp_dir, exist_ok=True)

        self.dataset_np = DatasetNP(args.datadir, args.dataname, self.conf)
        point_size = self.dataset_np.point_size
        self.dataname = args.dataname
        self.iter_step = 0

        # Training parameters
        self.maxiter = self.conf.get_int('train.maxiter')
        self.save_freq = self.conf.get_int('train.save_freq')
        self.size_update_freq = self.conf.get_int('train.size_update_freq')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_points_freq = self.conf.get_int('train.val_points_freq')
        self.val_mesh_freq = self.conf.get_int('train.val_mesh_freq')
        self.vertices_size = self.conf.get_int('train.vertices_size')
        self.update_size = self.conf.get_int('train.update_size')
        self.update_ratio = self.conf.get_float('train.update_ratio')
        self.k_samples = self.conf.get_int('train.k_samples')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)

        self.mode = mode
        self.generated_points = None

        # SDFNetwork
        sdf_network = SDFNetwork(point_size, **self.conf['model.sdf_network']).to(self.device)
        sdf_checkpoint = torch.load(os.path.join(self.sdf_exp_dir, 'sdf_checkpoints', args.sdf_checkpoint_name), map_location=self.device)
        self.sdf_iter_step = sdf_checkpoint['iter_step']
        sdf_network.load_state_dict(sdf_checkpoint['sdf_network_fine'])
        self.sdf_network = sdf_network
        # self.sdf_network = netutils.freeze(sdf_network) # do not use gradient

        # VGNetwork
        self.vg_network = VGNetwork_PTF(**self.conf['model.vg_network']).to(self.device)
        self.optimizer = torch.optim.Adam(self.vg_network.parameters(), lr=self.learning_rate)

        # Backup codes and configs for debug
        if self.mode[:5] == 'train':
            self.file_backup()

        if checkpoint_name:
            self.load_checkpoint(checkpoint_name)

    def train(self):
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = os.path.join(os.path.join(self.base_exp_dir), f'vg_{timestamp}.log')
        logger = get_root_logger(log_file=log_file, name='outs')
        self.logger = logger
        self.batch_size = self.generate_list_with_ratio()
        print("vertices generation ", self.batch_size)
        batch_size = self.batch_size[0]

        # cal curvature and normal
        point_gt = self.dataset_np.get_surface_queries(self.conf, self.sdf_network, self.sdf_iter_step)
        sample_points = self.dataset_np.fps_select_vertices(point_gt, batch_size)
        max_segment_length = 10 * 10000
        segments = torch.split(point_gt, max_segment_length)
        normal_list = []
        for segment in segments:
            normal_gt, _ = self.sdf_network.gradient(segment, step=self.sdf_iter_step)
            normal_gt, _ = F.normalize(normal_gt.detach(), dim=-1), _.detach()
            normal_list.append(normal_gt)
            torch.cuda.empty_cache()
        normal_gt = torch.cat(normal_list, dim=0)
        curvature_surface = losses.cal_curvature_with_normal(point_gt, normal_gt, self.conf.get_int('dataset.gt_curvature_knn')).detach()
        sample_normal, _ = self.sdf_network.gradient(sample_points, step=self.sdf_iter_step)
        sample_normal = F.normalize(sample_normal.detach(), dim=-1)
        torch.cuda.empty_cache()

        # visual
        curvature_dir = os.path.join(self.base_exp_dir, 'vg_curvature')
        os.makedirs(curvature_dir, exist_ok=True)
        name = os.path.join(curvature_dir, '{:0>8d}_surpts.ply'.format(self.iter_step))
        visualization.visible_points_curvature(point_gt.detach().cpu(), curvature_surface.detach().cpu(), name)
        self.validate_points(sample_points, move=False, real_world=True)
        self.generated_points = sample_points.detach().cpu().numpy()
        sdf_level = self.conf.get_float('dataset.project_sdf_level')
        self.validate_delaunay_mesh(sdf_level, self.k_samples, real_world=True)
        torch.cuda.empty_cache()

        res_step = self.maxiter - self.iter_step
        for iter_i in tqdm(range(res_step)):
            # self.update_learning_rate_np(iter_i)

            generated_vertices_ = self.vg_network(sample_points, sample_normal, self.iter_step)
            vertices_grad, _ = self.sdf_network.gradient(generated_vertices_, self.sdf_iter_step)

            loss = losses.cal_vg_loss(point_gt, normal_gt, curvature_surface, generated_vertices_, vertices_grad, self.conf)

            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()

            self.iter_step += 1
            if self.iter_step % self.report_freq == 0:
                print_log('iter:{:8>d} loss = {} lr={}'.format(self.iter_step, loss, self.optimizer.param_groups[0]['lr']), logger=logger)

            if self.iter_step % self.val_points_freq == 0:
                generated_vertices = self.move2surface(generated_vertices_, sdf_level)
                self.validate_points(generated_vertices_, move=False, real_world=True)
                self.validate_points(generated_vertices, move=True, real_world=True)

            if self.iter_step % self.val_mesh_freq == 0:
                self.generated_points = self.move2surface(generated_vertices_, sdf_level).detach().cpu().numpy()
                self.validate_delaunay_mesh(sdf_level, self.k_samples, real_world=True)

            if self.iter_step % self.save_freq == 0:
                self.save_checkpoint()

            if self.iter_step % self.size_update_freq == 0:
                it = self.iter_step // self.size_update_freq
                if it <= self.update_size:
                    batch_size = self.batch_size[it]
                    generated_vertices = self.move2surface(generated_vertices_, sdf_level)
                    curvature_sample = losses.cal_curvature_with_normal(generated_vertices, F.normalize(vertices_grad.detach(), dim=-1), self.conf.get_int('dataset.sample_curvature_knn'))
                    sample_points, sample_curvature = sampling.up_sampling(curvature_sample, generated_vertices, point_gt, batch_size)
                    sample_points = sample_points.detach()
                    sample_normal, _ = self.sdf_network.gradient(sample_points, step=self.sdf_iter_step)
                    sample_normal = F.normalize(sample_normal.detach(), dim=-1)
                    torch.cuda.empty_cache()
                    name = os.path.join(curvature_dir, '{:0>8d}_newadd.ply'.format(self.iter_step))
                    visualization.visible_points_curvature(sample_points.detach().cpu(), sample_curvature.detach().cpu(), name)

    def move2surface(self, generated_vertices, sdf_level, step=10):
        for i in range(step):
            gradients_queries, sdf_queries = self.sdf_network.gradient(generated_vertices, self.sdf_iter_step)
            gradients_queries, sdf_queries = gradients_queries.detach(), sdf_queries.detach()
            gradients_queries_norm = F.normalize(gradients_queries, dim=-1)
            queries_moved = generated_vertices - gradients_queries_norm * (sdf_queries - sdf_level)
            generated_vertices = queries_moved.detach()
            torch.cuda.empty_cache()

        return generated_vertices.detach()

    def validate_points(self, generated_vertices, move=True, real_world=True):
        vertices_dir = os.path.join(self.base_exp_dir, 'delaunay_vertices')
        os.makedirs(vertices_dir, exist_ok=True)
        pcd = trimesh.PointCloud(vertices=generated_vertices.detach().cpu().numpy())
        if real_world:
            pcd.apply_scale(self.dataset_np.scale)
            pcd.apply_translation(self.dataset_np.loc)

        name = '{:0>8d}_move.ply'.format(self.iter_step) if move else '{:0>8d}.ply'.format(self.iter_step)
        pcd.export(os.path.join(vertices_dir, name))

    def validate_fps(self, real_world=True):
        self.batch_size = self.generate_list_with_ratio()
        point_gt = self.dataset_np.get_surface_queries(self.conf, self.sdf_network, self.sdf_iter_step)
        fps_data = self.dataset_np.fps_select_vertices(point_gt, self.batch_size[-1])
        data_dir = os.path.join(self.base_exp_dir, 'vg_curvature')
        os.makedirs(data_dir, exist_ok=True)
        pcd = trimesh.PointCloud(vertices=fps_data.detach().cpu().numpy())
        if real_world:
            pcd.apply_scale(self.dataset_np.scale)
            pcd.apply_translation(self.dataset_np.loc)
        pcd.export(os.path.join(data_dir, 'fps_vertices.ply'))

    def validate_delaunay_mesh(self, sdf_level, k_samples, real_world=True):
        data_dir = os.path.join(self.base_exp_dir, 'delaunay_mesh')
        os.makedirs(data_dir, exist_ok=True)
        mesh_path = delaunay_meshing(data_dir, sdf_level, self.generated_points, self.sdf_network, self.iter_step, k_samples, self.sdf_iter_step, self.conf)
        if real_world:
            mesh = trimesh.load(mesh_path)
            mesh.apply_scale(self.dataset_np.scale)
            mesh.apply_translation(self.dataset_np.loc)
            mesh.export(mesh_path)

    def update_learning_rate_np(self, iter_step):
        warn_up = self.warm_up_end
        max_iter = self.maxiter
        init_lr = self.learning_rate
        lr = (iter_step / warn_up) if iter_step < warn_up else 0.5 * (math.cos((iter_step - warn_up)/(max_iter - warn_up) * math.pi) + 1)
        lr = lr * init_lr
        for g in self.optimizer.param_groups:
            g['lr'] = lr

    def generate_list_with_ratio(self):
        list = [int(self.vertices_size * 5 / 4.95)]
        for i in range(self.update_size):
            new = int(list[-1] / self.update_ratio) + 1
            list.append(new)
        list.reverse()
        return list

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
        # other files
        os.system("""cp *.py "{}" """.format(os.path.join(self.base_exp_dir, 'recording')))
        copyfile(self.conf_path, os.path.join(self.base_exp_dir, 'recording', 'vg_config.conf'))

    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'vg_checkpoints', checkpoint_name), map_location=self.device)
        print(os.path.join(self.base_exp_dir, 'vg_checkpoints', checkpoint_name))
        self.vg_network.load_state_dict(checkpoint['vg_network_fine'])
        self.iter_step = checkpoint['iter_step']
        self.generated_points = checkpoint['delaunay_vertices'].cpu().numpy()
            
    def save_checkpoint(self):
        checkpoint = {
            'vg_network_fine': self.vg_network.state_dict(),
            'iter_step': self.iter_step,
            'delaunay_vertices': torch.from_numpy(self.generated_points),
        }
        os.makedirs(os.path.join(self.base_exp_dir, 'vg_checkpoints'), exist_ok=True)
        torch.save(checkpoint, os.path.join(self.base_exp_dir, 'vg_checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))
    
        
if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/vg.conf')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--sdf_subdatadir', type=str, default='SDF')
    parser.add_argument('--sdf_checkpoint_name', type=str, default='ckpt_020000.pth')
    parser.add_argument('--datadir', type=str, default='./example/data/')
    parser.add_argument('--expdir', type=str, default='./example/exp/')
    parser.add_argument('--dataname', type=str, default='47984')
    parser.add_argument('--subdatadir', type=str, default='VG')
    parser.add_argument('--checkpoint_name', type=str, default=None)
    args = parser.parse_args()

    setup_seed(266815867)
    torch.cuda.set_device(args.gpu)
    runner = Runner(args, args.conf, args.mode, args.checkpoint_name)

    if args.mode == 'train':
        runner.train()
    elif args.mode == "validate_mesh_delaunay":
        threshs = [0.0]
        # threshs = [0.001]
        for thresh in threshs:
            generated_points = runner.move2surface(torch.tensor(runner.generated_points).cuda(), thresh, step=10)
            runner.generated_points = generated_points.detach().cpu().numpy()
            runner.validate_delaunay_mesh(thresh, k_samples=runner.k_samples, real_world=True)
    elif args.mode == "validate_fps":
        runner.validate_fps(real_world=True)