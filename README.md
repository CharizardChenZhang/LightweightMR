# High-Fidelity Lightweight Mesh Reconstruction from Point Clouds (CVPR 2025 Highlight)

PyTorch implementation of the paper:  

[High-Fidelity Lightweight Mesh Reconstruction from Point Clouds](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_High-Fidelity_Lightweight_Mesh_Reconstruction_from_Point_Clouds_CVPR_2025_paper.pdf) 

Chen Zhang, Wentao Wang, Ximeng Li, Xinyao Liao, Wanjuan Su, Wenbing Tao*

## Setup  
Python package requirements.  
```
python == 3.9
pytorch == 1.12.1
torch_scatter
trimesh
mcubes
open3d
fpsample
```
In addition, you also need to install some C++ libraries.  
```
Open3D == 0.16
CGAL
boost
libgmp-dev
```

## Compiling
Before starting, you need to compile several C++ programs.  
- Compile a kdtree algorithm for fast querying.
```
cd models/cpplib
conda activate your_env
CC=gcc CXX=gcc python setup.py build_ext --inplace
```
- Compile executables to construct Delaunay triangulation and generate the mesh.
```
cd models/delaunay_meshing

cd create_delaunay
# mkdir build
# cd build
# cmake ../
cmake ./
make

cd create_delaunay
# mkdir build
# cd build
# cmake ../
cmake ./
make
```

## Quick Test
There is some data in the `example` folder for reference.  
First, learn the SDF from the point cloud. `example/exp/xxx/SDF` is the output folder.
```
bash scripts/run_sdf.sh
```
Second, generate the mesh from point cloud and learned SDF. `example/exp/xxx/VG` is the output folder. The number of vertices can be specified by the parameter `vertices_size` in `confs/vg.conf`.
```
bash scripts/run_vg.sh
```

## Citation
```
@inproceedings{zhang2025high,
  title={High-Fidelity Lightweight Mesh Reconstruction from Point Clouds},
  author={Zhang, Chen and Wang, Wentao and Li, Ximeng and Liao, Xinyao and Su, Wanjuan and Tao, Wenbing},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={11739--11748},
  year={2025}
}
```
