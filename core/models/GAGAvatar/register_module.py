#!/usr/bin/env python
# Copyright (c) Sai Tanmay Reddy Chakkera (schakkera@cs.stonybrook.edu)

import torch
import torch.nn as nn

from core.libs.GAGAvatar_track.engines import project_3d_points_to_screen


class SmallerConvNetConvert(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SmallerConvNetConvert, self).__init__()
        self.net = nn.Sequential(
            # Layer 1: Reduce channels
            nn.Conv2d(in_channel, 512, kernel_size=3, stride=1, padding=1),  # -> (1024, 592, 592)
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # Layer 2
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),  # -> (1024, 592, 592)
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # Layer 3: Downsample
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),   # -> (512, 296, 296)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Layer 4
            nn.Conv2d(256, out_channel, kernel_size=3, stride=1, padding=1),    # -> (512, 296, 296)
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ThreeDRegisterNetwork(nn.Module):
    def __init__(self, embedding_size, num_vertices, topk=11, use_gaussian_noise_for_vertices=False, remove_proc_1=False):
        super().__init__()
        self.embedding_size = embedding_size
        self.num_vertices = num_vertices
        self.use_gaussian_noise_for_vertices = use_gaussian_noise_for_vertices

        self.embeddings = nn.Embedding(num_vertices+1, embedding_size)
        if use_gaussian_noise_for_vertices:
            self.embeddings.weight.requires_grad = False
        else:
            nn.init.xavier_normal_(self.embeddings.weight)
        self.cam_params = {'focal_x': 12.0, 'focal_y': 12.0, 'size': [296, 296]}
        self.filler_embedding = nn.Parameter(torch.randn(embedding_size), requires_grad=True)
        if remove_proc_1:
            self.convert_convnet = nn.Identity()
        else:
            self.convert_convnet = SmallerConvNetConvert(embedding_size, 256)
        self.output_processing = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1),
                                     nn.ReLU(True),
                                     nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1),
                                     nn.ReLU(True),
                                     nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1),
                                     nn.ReLU(True),
                                     nn.Conv2d(256, 256, kernel_size=1, padding=0, stride=1)
                                     )
        self.k = topk

    @staticmethod
    def interpolate_features_chunked(all_mat, proj_mat, feat_mat, k, chunk_size=200, topk_dists=None, topk_ids=None):
        B, M, D = all_mat.shape
        S = proj_mat.shape[1]
        F = feat_mat.shape[-1]

        device = all_mat.device
        result = torch.zeros(B, M, F, device=device)
        not_in_proj_mask = torch.zeros(B, M, dtype=torch.bool, device=device)

        if topk_ids is None:

            for start in range(0, M, chunk_size):
                end = min(start + chunk_size, M)

                All_chunk = all_mat[:, start:end, :]  # B x C x D

                # Pairwise distance: B x C x S
                dist_chunk = torch.cdist(All_chunk, proj_mat, p=2)

                # Equality check to detect points that already exist in proj_mat
                eq_chunk = (All_chunk[:, :, None, :] == proj_mat[:, None, :, :]).all(dim=-1)  # B x C x S
                in_proj_chunk = eq_chunk.any(dim=-1)  # B x C
                not_in_proj = ~in_proj_chunk  # B x C

                # Get k nearest neighbor indices
                topk_dists, topk_indices = torch.topk(dist_chunk, k, dim=-1, largest=False)

                # Gather features: B x C x k x F
                gathered_feats = torch.gather(
                    feat_mat[:, None, :, :].expand(-1, end - start, -1, -1),
                    dim=2,
                    index=topk_indices.unsqueeze(-1).expand(-1, -1, -1, F)
                )

                # Weights: reciprocal of distances
                weights = 1.0 / (topk_dists + 1e-8)
                weights = weights / weights.sum(dim=-1, keepdim=True)
                weights = weights.unsqueeze(-1)  # B x C x k x 1

                # Weighted sum
                interpolated_chunk = (weights * gathered_feats).sum(dim=2)  # B x C x F

                # Apply mask to only keep interpolated values where needed
                interpolated_chunk[~not_in_proj] = 0.0

                # Store results
                result[:, start:end, :] = interpolated_chunk
                not_in_proj_mask[:, start:end] = not_in_proj

            return result, not_in_proj_mask
        else:
            eq_all = (all_mat[:, :, None, :] == proj_mat[:, None, :, :]).all(dim=-1)
            in_proj = eq_all.any(dim=-1)
            not_in_proj = ~in_proj

            batch_indices = torch.arange(B, device=feat_mat.device).view(B, 1, 1).expand(-1, M, k)
            gathered_feats = feat_mat[batch_indices, topk_ids]
            weights = 1.0 / (topk_dists + 1e-8)
            weights = weights / weights.sum(dim=-1, keepdim=True)
            weights = weights.unsqueeze(-1)
            interpolated_feats = (weights * gathered_feats).sum(dim=2)
            interpolated_feats[~not_in_proj] = 0.0

            return interpolated_feats, not_in_proj


    def forward(self, features, target_points, visible_points, transform, all_pts_mask, topk_dists=None, topk_ids=None):
        target_points = torch.cat([target_points, torch.zeros(target_points.size(0), 1, 3, device=target_points.device)], dim=1) # bsize x 20018 x 3 -> bsize x 20019x 3

        # selecting the visible set of vertices given the indices of the vertices.
        visible_points_expanded = visible_points.unsqueeze(-1).expand(-1, -1, 3) # bsize x 20019 -> bsize x 20019 x 3
        visible_points_expanded = visible_points_expanded.to(torch.int64)
        visible_xyz = torch.gather(target_points, dim=1, index = visible_points_expanded) # bsize x 20019 x 3
        projected_points = project_3d_points_to_screen(visible_xyz, transform, 296)
        pixel_coords = torch.round(projected_points).int()

        #filter out any pixels outside the given size of the features.
        pixel_coords_mask = ((pixel_coords<0)|(pixel_coords>295)).any(dim=2)
        pixel_coords_invalid_ind = torch.nonzero(pixel_coords_mask)
        pixel_coords[pixel_coords_invalid_ind[:, 0], pixel_coords_invalid_ind[:, 1], :] = 0

        all_pts_mask_cond = ((all_pts_mask<0)|(all_pts_mask>295)).any(dim=2)
        all_pts_mask_invalid_ind = torch.nonzero(all_pts_mask_cond)
        all_pts_mask[all_pts_mask_invalid_ind[:, 0], all_pts_mask_invalid_ind[:, 1], :] = 0


        # query the embeddings dict to get the points that are visible
        tensor_of_features = self.embeddings(visible_points)

        # remove all the padding points in the set of visible points, implemented by replacing their embeddings with a set of 0.0 embeddings
        # IT SHOULD PROBABLY BE FILLER EMBEDDING INSTEAD OF A 0
        zero_mask = visible_points>=self.num_vertices
        zero_mask = zero_mask.unsqueeze(-1)
        zero_mask = zero_mask.expand(-1, -1, tensor_of_features.size(2))
        tensor_of_features = tensor_of_features.masked_fill(zero_mask, 0.0)
        tensor_of_features[pixel_coords_invalid_ind[:, 0], pixel_coords_invalid_ind[:, 1]] = self.filler_embedding

        interpolated_features, _ = self.interpolate_features_chunked(all_pts_mask.to(torch.float32), pixel_coords.to(torch.float32), tensor_of_features, self.k, topk_dists=topk_dists, topk_ids=topk_ids)


        # Construct an empty register output.
        register_output = torch.zeros(visible_points.size(0), *self.cam_params['size'], self.embedding_size, device=visible_xyz.device)

        # fill all rows with the filler embedding.
        register_output[:, :, :] = self.filler_embedding

        # fill the indices in the pixel coords with interpolated embeddings.
        bsize_a, seq_len_a = all_pts_mask.shape[:2]
        batch_idx_a = torch.arange(bsize_a).view(-1, 1).expand(-1, seq_len_a)
        x_all_pts_mask = all_pts_mask[:, :, 0]
        y_all_pts_mask = all_pts_mask[:, :, 1]
        register_output[batch_idx_a, y_all_pts_mask, x_all_pts_mask] = interpolated_features

        # fill the indices in the pixel coords with actual embeddings.
        bsize, seq_len = pixel_coords.shape[:2]
        batch_idx = torch.arange(bsize).view(-1, 1).expand(-1, seq_len)
        x_pixel_coords = pixel_coords[:, :, 0]
        y_pixel_coords = pixel_coords[:, :, 1]
        register_output[batch_idx, y_pixel_coords, x_pixel_coords] = tensor_of_features
        register_output[:, 0, 0] = self.filler_embedding
        output = self.convert_convnet(register_output.permute(0, 3, 1, 2))

        processed_output = self.output_processing(output+features)
        return processed_output, output, output+features, torch.sum(output)
