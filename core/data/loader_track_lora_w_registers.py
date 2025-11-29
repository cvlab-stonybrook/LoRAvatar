#!/usr/bin/env python
# Copyright (c) Sai Tanmay Reddy Chakkera (schakkera@cs.stonybrook.edu)


import os
import json
import pickle
import random
import torch
import trimesh
import torchvision
import numpy as np
from copy import deepcopy

from core.data.loader_track import build_points_planes
from core.libs.flame_model import FLAMEModel
from core.libs.utils_lmdb import LMDBEngine


class LoRAwRegistersDriverData(torch.utils.data.Dataset):
    def __init__(self, driver_path, feature_data=None, point_plane_size=296, num_vertices=20018):
        super().__init__()
        if type(driver_path) == str:
            self.driver_path = driver_path
            # build records
            self._is_video = True
            _records_path = os.path.join(self.driver_path, 'smoothed_np.pkl')
            if not os.path.exists(_records_path):
                _records_path = os.path.join(self.driver_path, 'smoothed.pkl')
            if not os.path.exists(_records_path):
                self._is_video = False
                _records_path = os.path.join(self.driver_path, 'optim.pkl')
            with open(_records_path, 'rb') as f:
                self._data = pickle.load(f)
                self._frames = sorted(list(self._data.keys()), key=lambda x: int(x.split('_')[-1]))
            if not self._is_video:
                self.shuffle_slice(60)
        else:
            self._is_video = False
            self._data = driver_path
            self._frames = list(self._data.keys())
            self._lmdb_engine = {key: self._data[key]['image'] * 255.0 for key in self._data.keys()}

        self.feature_data = feature_data
        self.point_plane_size = point_plane_size
        self._randomize = False
        # build model
        self.flame_model = FLAMEModel(n_shape=300, n_exp=100, scale=5.0, no_lmks=True)
        self.num_vertices = num_vertices
        # build feature data
        if feature_data is None:
            _lmdb_engine = LMDBEngine(os.path.join(self.driver_path, 'img_lmdb'), write=False)
            frame_key = random.choice(self._frames)
            _f_image = _lmdb_engine[frame_key].float() / 255.0
            self.f_image = torchvision.transforms.functional.resize(_f_image, (518, 518), antialias=True)
            f_transform = torch.tensor(self._data[frame_key]['transform_matrix']).float().cpu()
            self.f_planes = build_points_planes(self.point_plane_size, f_transform)
            self.f_shape = torch.tensor(self._data[frame_key]['shapecode']).float().cpu()
            _lmdb_engine.close()
        else:
            self.f_image = torchvision.transforms.functional.resize(self.feature_data['image'].cpu(), (518, 518),
                                                                    antialias=True)
            f_transform = self.feature_data['transform_matrix'].float().cpu()
            self.f_planes = build_points_planes(self.point_plane_size, f_transform)
            self.f_shape = self.feature_data['shapecode'].float().cpu()


    def slice(self, slice):
        self._frames = self._frames[:slice]

    def shuffle_slice(self, slice_num):
        import time
        import random
        random.seed(time.time())
        random.shuffle(self._frames)
        self._frames = self._frames[:slice_num]

    def __getitem__(self, index):
        frame_key = self._frames[index]
        return self._load_one_record(frame_key)

    def __len__(self, ):
        return len(self._frames)


    def _init_lmdb_database(self):
        # print('Init the LMDB Database!')
        self._lmdb_engine = LMDBEngine(os.path.join(self.driver_path, 'img_lmdb'), write=False)

    def _load_one_record(self, frame_key):
        if not hasattr(self, '_lmdb_engine'):
            self._init_lmdb_database()
        this_record = self._data[frame_key]
        for key in this_record.keys():
            if isinstance(this_record[key], np.ndarray):
                this_record[key] = torch.tensor(this_record[key])
        t_image = self._lmdb_engine[frame_key].float() / 255.0
        t_points = self.flame_model(
            shape_params=self.f_shape[None], pose_params=this_record['posecode'][None],
            expression_params=this_record['expcode'][None], eye_pose_params=this_record['eyecode'][None],
        )[0].float()
        mesh = trimesh.Trimesh(t_points.squeeze().numpy(), self.flame_model.get_faces().numpy())
        mesh = mesh.subdivide()
        vertices_sub = torch.from_numpy(np.array(mesh.vertices)).to(torch.float32)
        if 'all_pts_mask_sub' in this_record.keys():
            if 'topk_tuple_sub' in this_record.keys():

                assert this_record['all_pts_mask_sub'].size(0) == this_record['topk_tuple_sub'][0].size(1)
            one_data = {
                'f_image': deepcopy(self.f_image), 'f_planes': deepcopy(self.f_planes),
                't_planes': build_points_planes(self.point_plane_size,
                                                this_record['transform_matrix'].float().cpu()),
                't_image': t_image, 't_points': t_points, 't_transform': this_record['transform_matrix'],
                't_bbox': this_record['bbox'],
            }
            if 'topk_tuple_sub' in this_record.keys():
                one_data.update({'t_visible_verts': torch.cat([this_record['visible_verts'],
                                                                                 self.num_vertices * torch.ones(
                                                                                     self.num_vertices + 1 - len(
                                                                                         this_record['visible_verts']),
                                                                                     dtype=torch.int32)], 0),
                    't_visible_verts_sub': torch.cat((this_record['visible_verts_sub'],
                                                      self.num_vertices * torch.ones(
                                                          self.num_vertices + 1 - len(
                                                              this_record['visible_verts_sub']),
                                                          dtype=torch.int32)), 0),
                    't_vertices_sub': vertices_sub,
                    'all_pts_mask': torch.cat((this_record['all_pts_mask'],
                                               -1 * torch.ones(50000 - len(this_record['all_pts_mask']), 2,
                                                               dtype=torch.int32)), 0),
                    'all_pts_mask_sub': torch.cat((this_record['all_pts_mask_sub'],
                                                   -1 * torch.ones(50000 - len(this_record['all_pts_mask_sub']), 2,
                                                                   dtype=torch.int32)), 0),
                    'topk_dists': torch.cat((this_record['topk_tuple_sub'][1].squeeze(),
                                             torch.ones(50000 - len(this_record['all_pts_mask_sub']), 11,
                                                        dtype=torch.int32)), 0),
                    'topk_ids': torch.cat((this_record['topk_tuple_sub'][0].squeeze(),
                                           torch.ones(50000 - len(this_record['all_pts_mask_sub']), 11,
                                                      dtype=torch.int64)), 0)})
        else:
            if 'topk_tuple' in this_record.keys():
                assert this_record['all_pts_mask'].size(0) == this_record['topk_tuple'][1].size(1)
            one_data = {
                'f_image': deepcopy(self.f_image), 'f_planes': deepcopy(self.f_planes),
                't_planes': build_points_planes(self.point_plane_size,
                                                this_record['transform_matrix'].float().cpu()),
                't_image': t_image, 't_points': t_points, 't_transform': this_record['transform_matrix'],
                't_bbox': this_record['bbox'],
                'infos': {'t_key': frame_key},
            }
            if 'topk_tuple' in this_record.keys():
                one_data.update({'t_visible_verts': torch.cat((this_record['visible_verts'],
                                                                             self.num_vertices * torch.ones(
                                                                                 self.num_vertices + 1 - len(
                                                                                     this_record['visible_verts']),
                                                                                 dtype=torch.int32)), 0),
                'all_pts_mask': torch.cat((this_record['all_pts_mask'],
                                           -1 * torch.ones(50000 - len(this_record['all_pts_mask']), 2,
                                                           dtype=torch.int32)), 0),
                'topk_dists': torch.cat((this_record['topk_tuple'][1].squeeze(),
                                         torch.ones(50000 - len(this_record['topk_tuple'][1]), 11)), 0),
                'topk_ids': torch.cat((this_record['topk_tuple'][0].squeeze(),
                                       torch.ones(50000 - len(this_record['topk_tuple'][0]), 11, dype=torch.int64)),
                                      0)})
        return one_data

