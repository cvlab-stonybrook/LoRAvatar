import os
import torch
import argparse
import lightning
import numpy as np
import torchvision
from tqdm.rich import tqdm
import torch.nn as nn

from core.data import DriverData, LoRAwRegistersDriverData
from core.models import build_model
from core.libs.utils import ConfigDict
from core.libs.GAGAvatar_track.engines import CoreEngine as TrackEngine
from functools import partial
from minlora import add_lora, apply_to_lora, disable_lora, enable_lora, get_lora_params, merge_lora, name_is_lora, remove_lora, load_multiple_lora, select_lora
from minLoRA.minlora.model import LoRAParametrization, add_lora_by_name
from minLoRA.minlora.utils import get_lora_state_dict

permitted_parameters=['head_base_emb',
                      'gs_generator_g',
                      'gs_generator_l0',
                      'gs_generator_l1',
                      'upsampler',
                      ]

def lora_inference(image_path, lora_path, train_set,
                   driver_path, resume_path, force_retrack=False, threed_register=False,
                   threed_register_loss=False, gaussian_noise=False, learnable_noise=False,
                   remove_feat_loss=False, remove_reg_loss=False, use_gaussian_noise_for_vertices=False, remove_proc_1=False, device='cuda', lambda_reg=20.0):
    lightning.fabric.seed_everything(42)
    driver_path = driver_path[:-1] if driver_path.endswith('/') else driver_path
    driver_name = os.path.basename(driver_path).split('.')[0]
    # load model
    print(f'Loading model...')
    lightning_fabric = lightning.Fabric(accelerator=device, strategy='auto', devices=[0],)
    lightning_fabric.launch()
    full_checkpoint = lightning_fabric.load(resume_path)
    meta_cfg = ConfigDict(init_dict=full_checkpoint['meta_cfg'])
    model = build_model(model_cfg=meta_cfg.MODEL)
    model.load_state_dict(full_checkpoint['model'], strict=False)
    lora_rank=32
    lora_config = {
        nn.Embedding: {
            "weight": partial(LoRAParametrization.from_embedding, rank=lora_rank),
        },
        nn.Linear: {
            "weight": partial(LoRAParametrization.from_linear, rank=lora_rank),
        },
        nn.Conv2d: {
            "weight": partial(LoRAParametrization.from_conv2d, rank=lora_rank),
        },
        nn.ConvTranspose2d: {
            "weight": partial(LoRAParametrization.from_convtranspose2d, rank=lora_rank),
        },
    }
    for p in model.parameters():
        p.requires_grad_(False)
    add_lora_by_name(model, permitted_parameters, lora_config=lora_config)
    model_dir = os.path.join('render_results', os.path.basename(lora_path).split('.')[0])
    add_string = 'gaussian_noise_' if gaussian_noise else ""
    add_string = 'learnable_noise_' if learnable_noise else add_string
    add_string = "threed_register_" if threed_register else add_string
    add_string = add_string + 'w_register_loss_' if threed_register_loss else add_string + ''
    add_string = add_string + 'wo_feat_loss_' if remove_feat_loss else add_string
    add_string = add_string + 'wo_reg_loss_' if remove_reg_loss else add_string
    add_string = add_string + 'gaussian_noise_for_vertices_' if use_gaussian_noise_for_vertices else add_string
    add_string = add_string + 'remove_proc_1_' if remove_proc_1 else add_string
    add_string = add_string + f'lambda_reg_{lambda_reg}' if lambda_reg != 20.0 else add_string
    print("add_string", add_string)
    model_path = os.path.join(model_dir, f'lora_all_{add_string}'+train_set+'_'+str(max([int(file.split('_')[-1].split('.')[0]) for file in os.listdir(model_dir) if f'lora_all_{add_string}' in file and '.pt' in file]))+'.pt')
    print("loaded model at ", model_path)
    lora_weights=torch.load(model_path)
    model.load_state_dict(lora_weights['lora_weights'], strict=False)
    merge_lora(model)
    if threed_register_loss:
        print("Setting 3D Register Loss")
        model.set_3d_register_loss(lambda_reg=lambda_reg)
    if threed_register:
        print(f"Setting 3D Register")
        model.set_3d_register_network(use_gaussian_noise_for_vertices=use_gaussian_noise_for_vertices, remove_proc_1=remove_proc_1)
        model.register_network.load_state_dict(lora_weights['register_network'], strict=True)

    model = lightning_fabric.setup(model)
    print(str(meta_cfg))
    track_engine = TrackEngine(focal_length=12.0, device=device)
    # build input data
    feature_name = os.path.basename(image_path).split('.')[0]
    feature_data = get_tracked_results(image_path, track_engine, force_retrack=force_retrack)
    if feature_data is None:
        print(f'Finish inference, no face in input: {image_path}.')
        return
    # build driver data
    ### ------------ run on demo or tracked images/videos ---------- ###
    if os.path.isdir(driver_path):
        driver_name = os.path.basename(driver_path[:-1] if driver_path.endswith('/') else driver_path)
        driver_dataset = LoRAwRegistersDriverData(driver_path, feature_data, meta_cfg.DATASET.POINT_PLANE_SIZE)
        driver_dataloader = torch.utils.data.DataLoader(driver_dataset, batch_size=1, num_workers=4, shuffle=False)
    else:
        driver_name = os.path.basename(driver_path).split('.')[0]
        driver_data = get_tracked_results(driver_path, track_engine, force_retrack=force_retrack)
        if driver_data is None:
            print(f'Finish inference, no face in driver: {image_path}.')
            return
        driver_dataset = DriverData({driver_name: driver_data}, feature_data, meta_cfg.DATASET.POINT_PLANE_SIZE)
        driver_dataloader = torch.utils.data.DataLoader(driver_dataset, batch_size=10, num_workers=4, shuffle=False)
    driver_dataloader = lightning_fabric.setup_dataloaders(driver_dataloader)
    images = []

    for idx, batch in enumerate(tqdm(driver_dataloader)):
        render_results = model.forward_expression(batch)
        gt_rgb = render_results['t_image'].clamp(0, 1)
        pred_sr_rgb = render_results['sr_gen_image'].clamp(0, 1)
        visualize_rgbs = torch.cat([gt_rgb, pred_sr_rgb], dim=-1).detach()
        images.append(visualize_rgbs.cpu())
    dump_dir = os.path.join('render_results', meta_cfg.MODEL.NAME.split('_')[0])
    os.makedirs(dump_dir, exist_ok=True)
    if driver_dataset._is_video:
        dump_path = os.path.join(dump_dir, f'lora_all_{add_string}{train_set}_{driver_name}_{feature_name}.mp4')
        merged_images = torch.cat(images, dim=0)
        feature_images = torch.stack([feature_data['image']]*merged_images.shape[0])
        merged_images = torch.cat([feature_images, merged_images], dim=-1)
        merged_images = (merged_images * 255.0).to(torch.uint8).permute(0, 2, 3, 1)
        torchvision.io.write_video(dump_path, merged_images, fps=25.0)
    else:
        dump_path = os.path.join(dump_dir, f'lora_all_{add_string}{train_set}_{driver_name}_{feature_name}.jpg')
        merged_images = torchvision.utils.make_grid(images, nrow=5, padding=0)
        feature_images = torchvision.utils.make_grid([feature_data['image']]*(merged_images.shape[-2]//512), nrow=1, padding=0)
        merged_images = torch.cat([feature_images, merged_images], dim=-1)
        torchvision.utils.save_image(merged_images, dump_path)
    print(f'Finish inference: {dump_path}.')


def get_tracked_results(image_path, track_engine, force_retrack=False):
    if not is_image(image_path):
        print(f'Please input a image path, got {image_path}.')
        return None
    tracked_pt_path = 'render_results/tracked/tracked.pt'
    if not os.path.exists(tracked_pt_path):
        os.makedirs('render_results/tracked', exist_ok=True)
        torch.save({}, tracked_pt_path)
    tracked_data = torch.load(tracked_pt_path, weights_only=False)
    image_base = os.path.basename(image_path)
    if image_base in tracked_data and not force_retrack:
        print(f'Load tracking result from cache: {tracked_pt_path}.')
    else:
        print(f'Tracking {image_path}...')
        image = torchvision.io.read_image(image_path, mode=torchvision.io.ImageReadMode.RGB).float()
        feature_data = track_engine.track_image([image], [image_path])
        if feature_data is not None:
            feature_data = feature_data[image_path]
            torchvision.utils.save_image(
                torch.tensor(feature_data['vis_image']), 'render_results/tracked/{}.jpg'.format(image_base.split('.')[0])
            )
        else:
            print(f'No face detected in {image_path}.')
            return None
        tracked_data[image_base] = feature_data
        # track all images in this folder
        other_names = [i for i in os.listdir(os.path.dirname(image_path)) if is_image(i)]
        other_paths = [os.path.join(os.path.dirname(image_path), i) for i in other_names]
        if len(other_paths) <= 130:
            print('Track on all images in this folder to save time.')
            other_images = [torchvision.io.read_image(imp, mode=torchvision.io.ImageReadMode.RGB).float() for imp in other_paths]
            other_feature_data = track_engine.track_image(other_images, other_names)
            for key in other_feature_data:
                torchvision.utils.save_image(
                    torch.tensor(other_feature_data[key]['vis_image']), 'render_results/tracked/{}.jpg'.format(key.split('.')[0])
                )
            tracked_data.update(other_feature_data)
        # save tracking result
        torch.save(tracked_data, tracked_pt_path)
    feature_data = tracked_data[image_base]
    for key in list(feature_data.keys()):
        if isinstance(feature_data[key], np.ndarray):
            feature_data[key] = torch.tensor(feature_data[key])
    return feature_data


def is_image(image_path):
    extention_name = image_path.split('.')[-1].lower()
    return extention_name in ['jpg', 'png', 'jpeg']


def add_water_mark(image, water_mark):
    _water_mark_rgb = water_mark[None, :3]
    _water_mark_alpha = water_mark[None, 3:4].expand(-1, 3, -1, -1) * 0.8
    _mark_patch = image[..., -water_mark.shape[-2]:, -water_mark.shape[-1]:]
    _mark_patch = _mark_patch * (1-_water_mark_alpha) + _water_mark_rgb * _water_mark_alpha
    image[..., -water_mark.shape[-2]:, -water_mark.shape[-1]:] = _mark_patch
    return image


### ------- multi-view camera helper -------- ###
def build_camera(angle, ori_transforms=None, device='cuda'):
    from pytorch3d.renderer.cameras import look_at_view_transform
    if ori_transforms is None:
        distance = 9.3
    else:
        distance = ori_transforms[..., 3].square().sum(dim=-1).sqrt()[0].item() * 1.0
        device = ori_transforms.device
    print(f'Camera distance: {distance}, angle: {angle}.')
    R, T = look_at_view_transform(distance, 5, angle, device=device) # D, E, A
    rotate_trans = torch.cat([R, T[:, :, None]], dim=-1)
    return rotate_trans


### ------------ run speed test ------------- ###
def speed_test():
    driver_path = './demos/vfhq_driver'
    resume_path = './assets/GAGAvatar.pt'
    lightning.fabric.seed_everything(42)
    # load model
    print(f'Loading model...')
    lightning_fabric = lightning.Fabric(accelerator='cuda', strategy='auto', devices=[0],)
    lightning_fabric.launch()
    full_checkpoint = lightning_fabric.load(resume_path)
    meta_cfg = ConfigDict(init_dict=full_checkpoint['meta_cfg'])
    model = build_model(model_cfg=meta_cfg.MODEL)
    model.load_state_dict(full_checkpoint['model'])
    model = lightning_fabric.setup(model)
    print(str(meta_cfg))
    # build driver data
    driver_name = os.path.basename(driver_path[:-1] if driver_path.endswith('/') else driver_path)
    driver_dataset = DriverData(driver_path, None, meta_cfg.DATASET.POINT_PLANE_SIZE)
    driver_dataloader = torch.utils.data.DataLoader(driver_dataset, batch_size=1, num_workers=2, shuffle=False)
    driver_dataloader = lightning_fabric.setup_dataloaders(driver_dataloader)
    # run inference process
    for idx, batch in enumerate(tqdm(driver_dataloader)):
        render_results = model.forward_expression(batch)
        gt_rgb = render_results['t_image'].clamp(0, 1)
        pred_sr_rgb = render_results['sr_gen_image'].clamp(0, 1)
    print(f'Finish speed test.')
    # torchvision.utils.save_image([gt_rgb[0], pred_sr_rgb[0]], 'speed_test.jpg')


if __name__ == '__main__':
    import warnings
    from tqdm.std import TqdmExperimentalWarning
    warnings.simplefilter("ignore", category=TqdmExperimentalWarning, lineno=0, append=False)
    # build args
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', '-i', required=True, type=str)
    parser.add_argument('--lora_path', '-l', required=True, type=str)
    parser.add_argument('--driver_path', '-d', required=True, type=str)
    parser.add_argument('--train_set', '-t', required=True, type=str)
    parser.add_argument('--force_retrack', '-f', action='store_true')
    parser.add_argument('--resume_path', '-r', default='./assets/GAGAvatar.pt', type=str)
    parser.add_argument('--threed_register', action='store_true')
    parser.add_argument('--gaussian_noise', action='store_true')
    parser.add_argument('--learnable_noise', action='store_true')
    parser.add_argument('--threed_register_loss', action='store_true')
    parser.add_argument('--use_gaussian_noise_for_vertices', action='store_true')
    parser.add_argument('--remove_proc_1', action='store_true')
    parser.add_argument('--lambda_reg', type=float, default=20.0)
    parser.add_argument('--remove_feat_loss', help='remove the mse loss between target dino and reg module output',
                        action='store_true')
    parser.add_argument('--remove_reg_loss', help='remove the regularization loss between reg module embeddings',
                        action='store_true')
    #parser.add_argument('--resume_path', '-r', default='./outputs/GAGAvatar_FEED_front/Nov14_0941_wnchz/checkpoints/best_50000_0.857.pt', type=str)
    args = parser.parse_args()
    # launch
    torch.set_float32_matmul_precision('high')
    lora_inference(args.image_path, args.lora_path, args.train_set, args.driver_path, args.resume_path,
                   args.force_retrack, threed_register=args.threed_register, threed_register_loss=args.threed_register_loss,
                   gaussian_noise=args.gaussian_noise, learnable_noise=args.learnable_noise,
                   remove_feat_loss=args.remove_feat_loss, remove_reg_loss=args.remove_reg_loss,
                   use_gaussian_noise_for_vertices=args.use_gaussian_noise_for_vertices, remove_proc_1=args.remove_proc_1,
                   lambda_reg=args.lambda_reg)
