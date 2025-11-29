import os
import math
import torch
import torchvision
import trimesh
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from pytorch3d.renderer import PerspectiveCameras
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix

from engines.vgghead_detector import reproject_vertices
from engines.flame_model import FLAMEModel, RenderMesh
from engines import project_3d_points_to_screen
from shapely.geometry import Polygon, MultiLineString, Point
from shapely.ops import polygonize
from scipy.spatial import Delaunay
from multiprocessing import Pool, cpu_count

device = torch.device('cuda:0')

def _check_points_inside(args):
    points_chunk, polygon = args
    return np.array([polygon.contains(Point(x, y)) for x, y in points_chunk])


def integer_points_in_shapely_polygon_parallel(shapely_polygon: Polygon) -> torch.Tensor:
    minx, miny, maxx, maxy = shapely_polygon.bounds
    minx, miny = int(np.floor(minx)), int(np.floor(miny))
    maxx, maxy = int(np.ceil(maxx)), int(np.ceil(maxy))

    # Generate grid of integer coordinate points in bounding box
    xs, ys = np.meshgrid(np.arange(minx, maxx + 1), np.arange(miny, maxy + 1))
    all_points = np.stack([xs.ravel(), ys.ravel()], axis=1)

    # Split points into chunks for parallel processing
    n_cores = cpu_count()
    chunks = np.array_split(all_points, n_cores)
    args = [(chunk, shapely_polygon) for chunk in chunks]

    # Use multiprocessing pool
    with Pool(n_cores) as pool:
        results = pool.map(_check_points_inside, args)

    # Concatenate valid points
    mask = np.concatenate(results)
    inside_points = all_points[mask]

    return torch.tensor(inside_points, dtype=torch.int32).cpu()


def alpha_shape(points, alpha):
    """
    Compute the alpha shape (concave hull) of a set of 2D points.
    points: (N, 2) array
    alpha: alpha parameter
    Returns: shapely.geometry.Polygon object representing the contour
    """
    if len(points) < 4:
        return Polygon(points)

    tri = Delaunay(points)
    edges = set()
    for ia, ib, ic in tri.simplices:
        pa, pb, pc = points[ia], points[ib], points[ic]
        a = np.linalg.norm(pb - pa)
        b = np.linalg.norm(pc - pb)
        c = np.linalg.norm(pa - pc)
        s = (a + b + c) / 2.0
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        if area == 0:
            continue
        circum_r = a * b * c / (4.0 * area)
        if circum_r < 1.0 / alpha:
            edges.update([(ia, ib), (ib, ic), (ic, ia)])

    edge_lines = [(points[i], points[j]) for i, j in edges]
    m = MultiLineString(edge_lines)
    polygons = list(polygonize(m))
    return max(polygons, key=lambda p: p.area)  # largest area polygon


def get_mask_points(projected_vertices):
    pixel_coords = torch.round(projected_vertices.squeeze()).int()
    polygon = alpha_shape(pixel_coords, 0.065)
    correct_points = integer_points_in_shapely_polygon_parallel(polygon)
    return correct_points


def get_topk_visible_verts(all_pts, visible_points, k=11, chunk_size=5000):
    all_pts = all_pts.to('cuda')
    visible_points = visible_points.to('cuda')
    B, M, D = all_pts.shape
    S = visible_points.shape[1]
    device = all_pts.device
    result_id = torch.zeros(B, M, k, device=device, dtype=torch.int64)
    result_dist = torch.zeros(B, M, k, device=device, dtype=torch.int32)

    for start in range(0, M, chunk_size):
        end = min(start + chunk_size, M)

        All_chunk = all_pts[:, start:end, :]  # B x C x D

        # Pairwise distance: B x C x S
        dist_chunk = torch.cdist(All_chunk, visible_points, p=2)

        # Get k nearest neighbor indices
        topk_dists, topk_indices = torch.topk(dist_chunk, k, dim=-1, largest=False)

        result_dist[:, start:end, :] = topk_dists
        result_id[:, start:end, :] = topk_indices
    return result_id.cpu(), result_dist.cpu()


def get_pixels_from_vertex_id(verts_ids, verts, transform):
    visible_points_expanded = verts_ids.unsqueeze(0).unsqueeze(2).expand(-1, -1,
                                                                         3)  # bsize x 20019 -> bsize x 20019 x 3
    visible_xyz = torch.gather(verts, dim=1, index=visible_points_expanded)  # bsize x 20019 x 3
    projected_points = project_3d_points_to_screen(visible_xyz, transform, 296)

    pixel_coords = torch.round(projected_points).int()
    # filter out any pixels outside the given size of the features.
    pixel_coords_mask = ((pixel_coords < 0) | (pixel_coords > 295)).any(dim=2)
    pixel_coords_invalid_ind = torch.nonzero(pixel_coords_mask)
    pixel_coords[pixel_coords_invalid_ind[:, 0], pixel_coords_invalid_ind[:, 1], :] = 0
    return pixel_coords.to(torch.float32)


def revise_visble_vertices(path_to_pkl):
    with open(path_to_pkl, 'rb') as f:
        data = pickle.load(f)
    list_of_keys = list(data.keys())
    for key_ele in tqdm(list_of_keys):
        transform = data[key_ele]['transform_matrix']

        flame_model = FLAMEModel(n_shape=300, n_exp=100, scale=5.0).to(device)
        vertices, _ = flame_model(
            shape_params=torch.tensor(data[key_ele]['shapecode'])[None].to(device),
            expression_params=torch.tensor(data[key_ele]['expcode'])[None].to(device),
            pose_params=torch.tensor(data[key_ele]['posecode'])[None].float().to(device),
            eye_pose_params=torch.tensor(data[key_ele]['eyecode'])[None].float().to(device)
        )
        visible_faces_mask = np.all(
            np.isin(flame_model.get_faces().cpu().numpy(), list(data[key_ele]['visible_verts'])), axis=1)
        visible_faces = flame_model.get_faces().cpu().numpy()[visible_faces_mask]
        new_vertices, new_faces, index_dict = trimesh.remesh.subdivide(vertices.squeeze().cpu().numpy(),
                                                                       flame_model.get_faces().cpu().numpy(),
                                                                       return_index=True)
        new_vertices = torch.tensor(new_vertices, dtype=torch.float32)
        new_faces = torch.tensor(new_faces, dtype=torch.int32)

        list_of_new_faces = torch.stack(
            [torch.tensor(index_dict[key_idx], dtype=torch.int32) for key_idx in range(len(index_dict.keys()))])
        list_of_new_face_ids = list_of_new_faces[visible_faces, :]
        list_of_new_face_ids_flattened = torch.flatten(list_of_new_face_ids)
        list_of_visible_vertices = new_faces[list_of_new_face_ids_flattened, :]
        list_of_visible_vertices_flattened = torch.flatten(list_of_visible_vertices)
        visible_vertices_unique = torch.unique(list_of_visible_vertices_flattened, sorted=True)
        data[key_ele]['visible_verts_sub'] = visible_vertices_unique
        data[key_ele]['vertices_sub'] = new_vertices
        visible_vertices_xyz = vertices[:, data[key_ele]['visible_verts'], :]
        projected_visible_vertices = project_3d_points_to_screen(visible_vertices_xyz,
                                                                 torch.tensor(transform).to(torch.float32).unsqueeze(0).to(device),
                                                                 296)
        data[key_ele]['all_pts_mask'] = get_mask_points(projected_visible_vertices.cpu())  # C1 x 2
        visible_vertices_xyz_sub = new_vertices.unsqueeze(0)[:, visible_vertices_unique, :]
        projected_visible_vertices_sub = project_3d_points_to_screen(visible_vertices_xyz_sub.to(device),
                                                                     torch.tensor(transform).to(
                                                                         torch.float32).unsqueeze(0).to(device), 296)
        data[key_ele]['all_pts_mask_sub'] = get_mask_points(projected_visible_vertices_sub.cpu())  # C2 x 2
        data[key_ele]['topk_tuple'] = get_topk_visible_verts(
            data[key_ele]['all_pts_mask'].unsqueeze(0).to(torch.float32).to(device),
            get_pixels_from_vertex_id(torch.from_numpy(data[key_ele]['visible_verts']).to(torch.int64).to(device), vertices,
                                      torch.tensor(transform).to(torch.float32).unsqueeze(0).to(device)))
        data[key_ele]['topk_tuple_sub'] = get_topk_visible_verts(
            data[key_ele]['all_pts_mask_sub'].unsqueeze(0).to(torch.float32).to(device),
            get_pixels_from_vertex_id(data[key_ele]['visible_verts_sub'].to(torch.int64).to(device), new_vertices.unsqueeze(0).to(device),
                                      torch.tensor(transform).to(torch.float32).unsqueeze(0).to(device)))

    new_path = os.path.join(os.path.dirname(path_to_pkl),
                            os.path.splitext(os.path.basename(path_to_pkl))[0] + '_np.pkl')
    with open(new_path, 'wb') as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pickle_path', '-p', required=True, type=str)

    args = parser.parse_args()
    revise_visble_vertices(args.pickle_path)
