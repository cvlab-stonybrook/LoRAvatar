# Adapted from code by Xuangeng Chu (xg.chu@outlook.com)
# Written by Sai Tanmay Reddy Chakkera (schakkera@cs.stonybrook.edu)

import math
import torch
import torchvision
# import lpips
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.renderer.implicit.harmonic_embedding import HarmonicEmbedding

from core.models.GAGAvatar.register_module import ThreeDRegisterNetwork
from core.models.modules import DINOBase, StyleUNet
from core.libs.utils_renderer import render_gaussian
from core.libs.utils_perceptual import FacePerceptualLoss


class GAGAvatar(nn.Module):
    def __init__(self, model_cfg=None, **kwargs):
        super().__init__()
        self.base_model = DINOBase(output_dim=256)
        for param in self.base_model.dino_model.parameters():
            param.requires_grad = False
        # dir_encoder
        n_harmonic_dir = 4
        self.direnc_dim = n_harmonic_dir * 2 * 3 + 3
        self.harmo_encoder = HarmonicEmbedding(n_harmonic_dir)
        # pre_trained
        self.head_base = nn.Parameter(torch.randn(5023, 256), requires_grad=True)
        self.head_base_emb = nn.Embedding.from_pretrained(self.head_base)
        if model_cfg!=None:
            self.model_cfg = model_cfg

        self.gs_generator_g = LinearGSGenerator(in_dim=1024, dir_dim=self.direnc_dim)
        self.gs_generator_l0 = ConvGSGenerator(in_dim=256, dir_dim=self.direnc_dim)
        self.gs_generator_l1 = ConvGSGenerator(in_dim=256, dir_dim=self.direnc_dim)
        self.cam_params = {'focal_x': 12.0, 'focal_y': 12.0, 'size': [512, 512]}
        self.upsampler = StyleUNet(in_size=512, in_dim=32, out_dim=3, out_size=512)
        self.percep_loss = FacePerceptualLoss(loss_type='l1', weighted=True)
        self._fm_loss = False
        self._threed_register_loss = False

    def forward(self, batch):
        batch_size = batch['f_image'].shape[0]
        t_image, t_bbox = batch['t_image'], batch['t_bbox']
        f_image, f_planes = batch['f_image'], batch['f_planes']
        t_points, t_transform =  batch['t_points'], batch['t_transform']
        t_visible_verts = batch['t_visible_verts']
        all_pts_mask = batch['all_pts_mask']

        t_vertices_sub = None
        t_visible_verts_sub = None
        all_pts_mask_sub = None
        if 't_vertices_sub' in batch.keys():
            t_vertices_sub = batch['t_vertices_sub'].to(torch.int64)
            t_visible_verts_sub = batch['t_visible_verts_sub'].to(torch.int64)
            all_pts_mask_sub = batch['all_pts_mask_sub']

        # feature encoding
        output_size = int(math.sqrt(f_planes['plane_points'].shape[1]))

        f_feature0, f_feature1 = self.base_model(f_image, output_size=output_size)
        if hasattr(self, 'register_network'):
            if self._threed_register_loss:
                f_feature0, _, _, output_sum = self.register_network(f_feature0,
                                                                     t_points if t_vertices_sub is None else t_vertices_sub,
                                                                     t_visible_verts if t_visible_verts_sub is None else t_visible_verts_sub, t_transform,
                                                                     all_pts_mask if all_pts_mask_sub is None else all_pts_mask_sub, topk_dists=batch['topk_dists'], topk_ids=batch['topk_ids'])
                target_f_feature0, target_f_feature1 = self.base_model(torchvision.transforms.functional.resize(t_image, (518, 518)),
                                                       output_size=output_size)

        if hasattr(self, 'gaussian_noise'):
            if self.gaussian_noise:
                f_feature0 += torch.randn(*f_feature0.size(), device=f_feature0.device)

        if hasattr(self, 'learnable_noise'):
            if self.learnable_noise:
                f_feature0 += torch.stack([self.learnable_embeddings]*f_feature0.size(0), dim=0)

        plane_direnc = self.harmo_encoder(f_planes['plane_dirs'])
        gs_params_g = self.gs_generator_g(
            torch.cat([
                self.head_base_emb(torch.arange(5023, dtype=torch.long, device=f_feature0.device))[None].expand(
                    batch_size, -1, -1), f_feature1[:, None].expand(-1, 5023, -1),
            ], dim=-1
            ), plane_direnc
        )
        gs_params_g['xyz'] = t_points
        # local part

        gs_params_l0 = self.gs_generator_l0(f_feature0, plane_direnc)
        gs_params_l1 = self.gs_generator_l1(f_feature0, plane_direnc)
        gs_params_l0['xyz'] = f_planes['plane_points'] + gs_params_l0['positions'] * f_planes['plane_dirs'][:, None]
        gs_params_l1['xyz'] = f_planes['plane_points'] + -1 * gs_params_l1['positions'] * f_planes['plane_dirs'][:, None]
        gs_params = {
            k:torch.cat([gs_params_g[k], gs_params_l0[k], gs_params_l1[k]], dim=1) for k in gs_params_g.keys()
        }
        gen_images = render_gaussian(
            gs_params=gs_params, cam_matrix=t_transform, cam_params=self.cam_params
        )['images']
        sr_gen_images = self.upsampler(gen_images)
        results = {
            'f_image': f_image,
            't_image':t_image, 't_bbox':t_bbox, 't_points': t_points, 
            'p_points': torch.cat([gs_params_l0['xyz'], gs_params_l1['xyz']], dim=1),
            'gen_image': gen_images[:, :3], 'sr_gen_image': sr_gen_images
        }
        if hasattr(self, 'register_network'):
            if self._threed_register_loss:
                results_addendum = {'target_f_feature0': target_f_feature0, 'f_feature0': f_feature0, 'output_sum':output_sum}
                results.update(results_addendum)
        return results

    def set_remove_feat_loss(self, remove_feat_loss=True):
        self.remove_feat_loss = remove_feat_loss

    def set_remove_reg_loss(self, remove_reg_loss=True):
        self.remove_reg_loss = remove_reg_loss

    def use_learnable_noise(self, learnable_noise=True):
        self.learnable_noise = learnable_noise
        self.learnable_embeddings = nn.Parameter(torch.randn(256, 296, 296), requires_grad=True)

    def use_gaussian_noise(self, gaussian_noise=True):
        self.gaussian_noise = gaussian_noise

    def set_3d_register_network(self, embedding_size=512, num_vertices=20018, use_gaussian_noise_for_vertices=False, remove_proc_1=False):
        if hasattr(self, 'model_cfg'):
            try:
                self.register_network = ThreeDRegisterNetwork(self.model_cfg.EMBEDDING_SIZE if not remove_proc_1 else 256, num_vertices=num_vertices, use_gaussian_noise_for_vertices=use_gaussian_noise_for_vertices, remove_proc_1=remove_proc_1)
            except:
                self.register_network = ThreeDRegisterNetwork(embedding_size if not remove_proc_1 else 256, num_vertices=num_vertices, use_gaussian_noise_for_vertices=use_gaussian_noise_for_vertices, remove_proc_1=remove_proc_1)
        else:
                self.register_network = ThreeDRegisterNetwork(embedding_size if not remove_proc_1 else 256, num_vertices=num_vertices, use_gaussian_noise_for_vertices=use_gaussian_noise_for_vertices, remove_proc_1=remove_proc_1)

    def set_3d_register_loss(self, threed_register_loss=True, lambda_reg=20.0):
        self._threed_register_loss = threed_register_loss
        self.lambda_reg = lambda_reg


    @torch.no_grad()
    def forward_expression(self, batch):
        if not hasattr(self, '_gs_params'):
            batch_size = batch['f_image'].shape[0]
            f_image, f_planes = batch['f_image'], batch['f_planes']
            f_feature0, f_feature1 = self.base_model(f_image)

            # dir encoding
            plane_direnc = self.harmo_encoder(f_planes['plane_dirs'])
            # global part
            gs_params_g = self.gs_generator_g(
                torch.cat([
                        self.head_base_emb(torch.arange(5023, dtype=torch.long, device=f_feature0.device))[None].expand(batch_size, -1, -1), f_feature1[:, None].expand(-1, 5023, -1),
                    ], dim=-1
                ), plane_direnc
            )
            gs_params_g['xyz'] = batch['f_image'].new_zeros((batch_size, 5023, 3))
            # local part
            gs_params_l0 = self.gs_generator_l0(f_feature0, plane_direnc)
            gs_params_l1 = self.gs_generator_l1(f_feature0, plane_direnc)
            gs_params_l0['xyz'] = f_planes['plane_points'] + gs_params_l0['positions'] * f_planes['plane_dirs'][:, None]
            gs_params_l1['xyz'] = f_planes['plane_points'] + -1 * gs_params_l1['positions'] * f_planes['plane_dirs'][:, None]
            gs_params = {
                k:torch.cat([gs_params_g[k], gs_params_l0[k], gs_params_l1[k]], dim=1) for k in gs_params_g.keys()
            }

            self._gs_params = gs_params
        gs_params = self._gs_params
        t_image, t_points, t_transform = batch['t_image'], batch['t_points'], batch['t_transform']

        gs_params['xyz'][:, :5023] = t_points
        gen_images = render_gaussian(
            gs_params=gs_params, cam_matrix=t_transform, cam_params=self.cam_params
        )['images']
        sr_gen_images = self.upsampler(gen_images)
        results = {
            't_image':t_image, 'gen_image': gen_images[:, :3], 'sr_gen_image': sr_gen_images,
        }
        return results

    @staticmethod
    def calc_self_cos_sim_mat(a):
        numer = torch.mm(a, a.T)
        denom = torch.mm(a.norm(dim=1)[:, None], a.norm(dim=1)[:, None].T)
        denom = torch.where(denom < 1e-8, 1e-8, denom)
        return torch.sum(numer/denom)

    def calc_metrics(self, results):
        loss_fn = nn.functional.l1_loss
        t_image, t_bbox = results['t_image'], results['t_bbox']
        t_bbox = expand_bbox(t_bbox, scale=1.1)

        gen_image, sr_gen_image = results['gen_image'], results['sr_gen_image']


        if self._threed_register_loss:
            register_reg_loss = (self.calc_self_cos_sim_mat(self.register_network.embeddings(
                torch.arange(self.register_network.num_vertices).to(
                    results['f_feature0'].device))) - self.register_network.num_vertices) / (
                                     self.register_network.num_vertices * (self.register_network.num_vertices - 1))


            f_feature0_loss = loss_fn(results['target_f_feature0'], results['f_feature0'])

            if hasattr(self, 'remove_feat_loss'):
                if self.remove_feat_loss:
                    f_feature0_loss = 0.
            if hasattr(self, 'remove_reg_loss'):
                if self.remove_reg_loss:
                    register_reg_loss = 0.

        img_loss_0 = (loss_fn(gen_image, t_image))
        img_loss_1 = (loss_fn(sr_gen_image, t_image))
        box_loss_0, bpec_loss_0 = self.calc_box_loss(gen_image, t_image, t_bbox, loss_fn)
        box_loss_1, bpec_loss_1 = self.calc_box_loss(sr_gen_image, t_image, t_bbox, loss_fn)
        pec_loss = (bpec_loss_0 + bpec_loss_1)
        img_loss = (img_loss_0 + img_loss_1) * 0.5
        box_loss = (box_loss_0 + box_loss_1) * 0.5
        point_loss = square_distance(results['t_points'], results['p_points']).mean()
        loss = {'percep_loss': pec_loss, 'img_loss': 3*img_loss, 'box_loss': box_loss, 'point_loss': point_loss}
        if self._threed_register_loss:
            loss['register_reg'] = self.lambda_reg*register_reg_loss
            loss['f_feature0_loss'] = 2*f_feature0_loss
        psnr = -10.0 * torch.log10(nn.functional.mse_loss(t_image, sr_gen_image).detach())
        return loss, {'psnr':psnr.item()}

    def configure_optimizers(self, config):
        learning_rate = config.LEARNING_RATE
        print('Learning rate: {}'.format(learning_rate))
        # params
        decay_names = []
        normal_params, decay_params0, decay_params1 = [], [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if 'style_mlp' in name or 'final_linear' in name:
                decay_names.append(".".join(name.split('.')[:-2]) if len(name.split('.'))> 3 else ".".join(name.split('.')[:-1]))
                decay_params0.append(param)
            elif 'gaussian_conv' in name or ('gs_generator_g' in name and 'feature_layers' not in name):
                # decay_names.append(".".join(name.split('.')[:-2]) if len(name.split('.'))> 3 else ".".join(name.split('.')[:-1]))
                decay_params1.append(param)
            else:
                normal_params.append(param)
        print('Decay params: {}'.format(set(decay_names)))
        # optimizer
        optimizer = torch.optim.Adam([
                {'params': normal_params, 'lr': learning_rate},
                {'params': decay_params0, 'lr': learning_rate*0.1},
                {'params': decay_params1, 'lr': learning_rate},
            ], lr=learning_rate, betas=(0.0, 0.99)
        )
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=config.LR_DECAY_RATE, 
            total_iters=config.LR_DECAY_ITER,
            # verbose=True if self.logger is None else False
        )
        return optimizer, scheduler

    def calc_box_loss(self, image, gt_image, bbox, loss_fn, resize_size=512):
        def _resize(frames, tgt_size):
            frames = nn.functional.interpolate(
                frames, size=(tgt_size, tgt_size), mode='bilinear', align_corners=False, antialias=True
            )
            return frames
        bbox = bbox.clamp(min=0, max=1)
        bbox = (bbox * image.shape[-1]).long()
        pred_croped, gt_croped = [], []
        for idx, box in enumerate(bbox):
            gt_croped.append(_resize(gt_image[idx:idx+1, :, box[1]:box[3], box[0]:box[2]], resize_size))
            pred_croped.append(_resize(image[idx:idx+1, :, box[1]:box[3], box[0]:box[2]], resize_size))
        gt_croped = torch.cat(gt_croped, dim=0)
        pred_croped = torch.cat(pred_croped, dim=0)
        # gt_croped_for_box_loss = gt_croped[:, :2] / (torch.sum(gt_croped, 1).view(gt_croped.size(0), 1, gt_croped.size(2), gt_croped.size(3)) + 1.0e-8)
        # pred_croped_for_box_loss = pred_croped[:, :2] / (torch.sum(pred_croped, 1).view(pred_croped.size(0), 1, pred_croped.size(2), pred_croped.size(3)) + 1.0e-8)

        box_fn_loss = (loss_fn(pred_croped, gt_croped)
                       # +loss_fn(self.get_hsv(pred_croped)[:, :1], self.get_hsv(gt_croped)[:, :1])
                       # +loss_fn(self.get_hsv(pred_croped)[:, 1:2], self.get_hsv(gt_croped)[:, 1:2])
                       # +loss_fn(self.get_hsv(pred_croped)[:, 2:3], self.get_hsv(gt_croped)[:, 2:3])
                       )
        # box_fn_loss = loss_fn(pred_croped, gt_croped)
        #box_fn_loss = loss_fn((1-pred_croped), (1-gt_croped))
        box_perc_loss = self.percep_loss(pred_croped, gt_croped) * 1e-2
        # box_loss = (box_1_loss + box_2_loss) / 2
        return box_fn_loss, box_perc_loss


class LinearGSGenerator(nn.Module):
    def __init__(self, in_dim=1024, dir_dim=27, **kwargs):
        super().__init__()
        # params
        self.feature_layers = nn.Sequential(
            nn.Linear(in_dim, in_dim//4, bias=True),
            nn.ReLU(),
            nn.Linear(in_dim//4, in_dim//4, bias=True),
            nn.ReLU(),
            nn.Linear(in_dim//4, in_dim//4, bias=True),
            nn.ReLU(),
            nn.Linear(in_dim//4, in_dim//4, bias=True),
        )
        layer_in_dim = in_dim//4 + dir_dim
        self.color_layers = nn.Sequential(
            nn.Linear(layer_in_dim, 128, bias=True),
            nn.ReLU(),
            nn.Linear(128, 32, bias=True),
        )
        self.opacity_layers = nn.Sequential(
            nn.Linear(layer_in_dim, 128, bias=True),
            nn.ReLU(),
            nn.Linear(128, 1, bias=True),
        )
        self.scale_layers = nn.Sequential(
            nn.Linear(layer_in_dim, 128, bias=True),
            nn.ReLU(),
            nn.Linear(128, 3, bias=True)
        )
        self.rotation_layers = nn.Sequential(
            nn.Linear(layer_in_dim, 128, bias=True),
            nn.ReLU(),
            nn.Linear(128, 4, bias=True),
        )

    def forward(self, input_features, plane_direnc):
        input_features = self.feature_layers(input_features)
        plane_direnc = plane_direnc[:, None].expand(-1, input_features.shape[1], -1)
        input_features = torch.cat([input_features, plane_direnc], dim=-1)
        # color
        colors = self.color_layers(input_features)
        colors[..., :3] = torch.sigmoid(colors[..., :3])
        # opacity
        opacities = self.opacity_layers(input_features)
        opacities = torch.sigmoid(opacities)
        # scale
        scales = self.scale_layers(input_features)
        # scales = torch.exp(scales) * 0.01
        scales = torch.sigmoid(scales) * 0.05
        # rotation
        rotations = self.rotation_layers(input_features)
        rotations = nn.functional.normalize(rotations)
        return {'colors':colors, 'opacities':opacities, 'scales':scales, 'rotations':rotations}


class ConvGSGenerator(nn.Module):
    def __init__(self, in_dim=256, dir_dim=27, **kwargs):
        super().__init__()
        out_dim = 32 + 1 + 3 + 4 + 1 # color + opacity + scale + rotation + position
        self.gaussian_conv = nn.Sequential(
            nn.Conv2d(in_dim+dir_dim, in_dim//2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_dim//2, in_dim//2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_dim//2, in_dim//2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_dim//2, out_dim, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, input_features, plane_direnc):
        plane_direnc = plane_direnc[:, :, None, None].expand(-1, -1, input_features.shape[2], input_features.shape[3])
        input_features = torch.cat([input_features, plane_direnc], dim=1)
        gaussian_params = self.gaussian_conv(input_features)
        # color
        colors = gaussian_params[:, :32]
        colors[..., :3] = torch.sigmoid(colors[..., :3])
        # opacity
        opacities = gaussian_params[:, 32:33]
        opacities = torch.sigmoid(opacities)
        # scale
        scales = gaussian_params[:, 33:36]
        # scales = torch.exp(scales) * 0.01
        scales = torch.sigmoid(scales) * 0.05
        # rotation
        rotations = gaussian_params[:, 36:40]
        rotations = nn.functional.normalize(rotations)
        # position
        positions = gaussian_params[:, 40:41]
        positions = torch.sigmoid(positions)
        results = {'colors':colors, 'opacities':opacities, 'scales':scales, 'rotations':rotations, 'positions':positions}
        for key in results.keys():
            results[key] = results[key].permute(0, 2, 3, 1).reshape(results[key].shape[0], -1, results[key].shape[1])
        return results


def square_distance(src, dst):
    import faiss
    assert src.dim() == 3 and dst.dim() == 3, 'Input tensors must be 3-dim.'
    all_indices = []
    for bid in range(src.shape[0]):
        src_np = src[bid].detach().cpu().numpy()
        dst_np = dst[bid].detach().cpu().numpy()
        index = faiss.IndexFlatL2(3)
        index.add(dst_np)
        _, indices = index.search(src_np, 1)
        all_indices.append(torch.tensor(indices))
    indices = torch.stack(all_indices).to(src.device)
    dst_selected = torch.gather(dst, 1, indices.to(src.device).expand(-1, -1, dst.shape[-1]))
    distances = torch.sum((src - dst_selected) ** 2, dim=-1) * 10
    return distances


def expand_bbox(bbox, scale=1.1):
    xmin, ymin, xmax, ymax = bbox.unbind(dim=-1)
    cenx, ceny = (xmin + xmax) / 2, (ymin + ymax) / 2
    extend_size = torch.sqrt((ymax - ymin) * (xmax - xmin)) * scale
    extend_size = torch.min(extend_size, cenx*2)
    extend_size = torch.min(extend_size, ceny*2)
    extend_size = torch.min(extend_size, (1-cenx)*2)
    extend_size = torch.min(extend_size, (1-ceny)*2)
    xmine, xmaxe = cenx - extend_size / 2, cenx + extend_size / 2
    ymine, ymaxe = ceny - extend_size / 2, ceny + extend_size / 2
    expanded_bbox = torch.stack([xmine, ymine, xmaxe, ymaxe], dim=-1)
    return torch.stack([xmine, ymine, xmaxe, ymaxe], dim=-1)
