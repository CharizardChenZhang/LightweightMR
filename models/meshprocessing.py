import trimesh
import numpy as np

def add_grid_pts(points, grid_range, grid_res):
    x, y, z = np.linspace(-grid_range, grid_range, grid_res).astype(np.float32), \
        np.linspace(-grid_range, grid_range, grid_res).astype(np.float32), \
        np.linspace(-grid_range, grid_range, grid_res).astype(np.float32)
    xx, yy, zz = np.meshgrid(x, y, z)
    xx, yy, zz = xx.ravel(), yy.ravel(), zz.ravel()
    grid_points = np.stack([xx, yy, zz], axis=1).astype('f')
    points = np.concatenate([points, grid_points], axis=0)

    return points

def separate_vertices(mesh):
    ordered_edges = np.sort(mesh.edges, axis=1)
    uni, count = np.unique(ordered_edges, return_counts=True, axis=0)
    idx = np.argwhere(count == 4).reshape(-1,)
    candidate_edges = uni[idx]

    new_vertices_list = []
    faces_list = []
    mesh_faces = np.array(mesh.faces)
    mesh_vertices = np.array(mesh.vertices)
    for i, edge in enumerate(candidate_edges):
        new_vertices = np.array(mesh.vertices[edge])
        matches = np.all(ordered_edges == edge, axis=1)
        indices = np.where(matches)[0]
        face_indices = indices // 3
        new_vertices_list.append(new_vertices)
        faces_list.append(face_indices)

    for i, edge in enumerate(candidate_edges):
        for j, face_idx in enumerate(np.split(faces_list[i], 2)):
            if j == 0:
                continue
            select_faces = mesh_faces[face_idx]    # 2,3
            mesh_vertices = np.concatenate([mesh_vertices, new_vertices_list[i]], axis=0)
            select_faces = np.where(select_faces == edge[0], len(mesh_vertices)-2, select_faces)
            select_faces = np.where(select_faces == edge[1], len(mesh_vertices)-1, select_faces)
            mesh_faces[face_idx] = select_faces

            face0_v = (set(select_faces[0]) - {len(mesh_vertices) - 2, len(mesh_vertices) - 1}).pop()
            face1_v = (set(select_faces[1]) - {len(mesh_vertices) - 2, len(mesh_vertices) - 1}).pop()
            adj0_edge0 = np.sort(np.array([face0_v, edge[0]]))
            adj0_edge1 = np.sort(np.array([face0_v, edge[1]]))
            adj1_edge0 = np.sort(np.array([face1_v, edge[0]]))
            adj1_edge1 = np.sort(np.array([face1_v, edge[1]]))
            adj_edges = [adj0_edge0, adj0_edge1, adj1_edge0, adj1_edge1]
            vert = [edge[0], edge[1], edge[0], edge[1]]
            values = [len(mesh_vertices)-2, len(mesh_vertices)-1, len(mesh_vertices)-2, len(mesh_vertices)-1]
            for k, adj_edge in enumerate(adj_edges):
                matches = np.all(ordered_edges == adj_edge, axis=1)
                adj_indices = np.where(matches)[0]
                adj_face_idx = adj_indices // 3
                adj_select_faces = mesh_faces[adj_face_idx]
                adj_select_faces = np.where(adj_select_faces == vert[k], values[k], adj_select_faces)
                mesh_faces[adj_face_idx] = adj_select_faces

    new_mesh = trimesh.Trimesh(vertices=mesh_vertices, faces=mesh_faces, process=False)

    return new_mesh

def merge_vertices(mesh):
    ordered_edges = np.sort(mesh.edges, axis=1)
    uni, count = np.unique(ordered_edges, return_counts=True, axis=0)
    idx = np.argwhere(count != 2).reshape(-1,)
    candidate_edges = uni[idx]    # n,2
    mesh_vertices = np.array(mesh.vertices)
    mesh_faces = np.array(mesh.faces)
    mesh_vertices[candidate_edges[:, 1]] = mesh_vertices[candidate_edges[:, 0]]

    new_mesh = trimesh.Trimesh(vertices=mesh_vertices, faces=mesh_faces)
    new_mesh.update_faces(new_mesh.nondegenerate_faces())

    return new_mesh

def add_mid_vertices(mesh):
    ordered_edges = np.sort(mesh.edges, axis=1)
    uni, count = np.unique(ordered_edges, return_counts=True, axis=0)
    idx = np.argwhere(count != 2).reshape(-1,)
    candidate_edges = uni[idx]

    new_vertices_list = []
    faces_list = []
    mesh_faces = np.array(mesh.faces)
    mesh_vertices = np.array(mesh.vertices)
    for i, edge in enumerate(candidate_edges):
        new_vertices = np.mean(mesh.vertices[edge], axis=0, keepdims=True)  # 1,3
        matches = np.all(ordered_edges == edge, axis=1)
        indices = np.where(matches)[0]
        face_indices = indices // 3
        new_vertices_list.append(new_vertices)
        faces_list.append(face_indices)

    for i, edge in enumerate(candidate_edges):
        for j, face_idx in enumerate(np.split(faces_list[i], 2)):
            if j == 0:
                continue
            select_faces = mesh_faces[face_idx]    # 2,3
            mesh_vertices = np.concatenate([mesh_vertices, new_vertices_list[i]], axis=0)
            select_faces_1 = np.where(select_faces == edge[0], len(mesh_vertices)-1, select_faces)
            select_faces_2 = np.where(select_faces == edge[1], len(mesh_vertices)-1, select_faces)
            mesh_faces[face_idx] = select_faces_1
            mesh_faces = np.concatenate([mesh_faces, select_faces_2], axis=0)

    new_mesh = trimesh.Trimesh(vertices=mesh_vertices, faces=mesh_faces, process=False)

    return new_mesh