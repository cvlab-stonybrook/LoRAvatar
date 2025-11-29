import os

import wandb
import torch
import argparse
import lightning
import numpy as np
import torchvision
import torch.nn as nn
from tqdm.rich import tqdm
from torch.optim import Adam

from core.data import DriverData, LoRAwRegistersDriverData
from core.models import build_model
from core.libs.utils import ConfigDict
from core.libs.GAGAvatar_track.engines import CoreEngine as TrackEngine

from functools import partial
from minlora import add_lora, apply_to_lora, disable_lora, enable_lora, get_lora_params, merge_lora, name_is_lora, remove_lora, load_multiple_lora, select_lora

from minLoRA.minlora.model import LoRAParametrization, add_lora_by_name
from minLoRA.minlora.utils import get_lora_state_dict
torch.set_warn_always(False)

permitted_parameters=['head_base_emb',
                      'gs_generator_g',
                      'gs_generator_l0',
                      'gs_generator_l1',
                      'upsampler',
                      ]

def lora_adaptation(image_path, driver_path, resume_path,
                    force_retrack=False, threed_register=False, threed_register_loss=False, device='cuda',
                    gaussian_noise=False, learnable_noise=False, remove_feat_loss=False, remove_reg_loss=False,
                    use_gaussian_noise_for_vertices=False, remove_proc_1=False, lambda_reg=20.0,
                    num_iters=1000):
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
    wandb.init(
        # set the wandb project where this run will be logged
        project="LoRAwRegistersAvatar",

        # track hyperparameters and run metadata
        config=meta_cfg._dump
    )

    if threed_register_loss:
        print("Setting 3D Register Loss")
        model.set_3d_register_loss(lambda_reg=lambda_reg)
    if threed_register:
        print("Setting 3D Register")
        model.set_3d_register_network(use_gaussian_noise_for_vertices=use_gaussian_noise_for_vertices, remove_proc_1=remove_proc_1)

    if remove_feat_loss:
        print("Removing Feat MSE Loss")
        model.set_remove_feat_loss()

    if remove_reg_loss:
        print("Removing Reg Loss")
        model.set_remove_reg_loss()

    lora_rank = 32
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
    for n, p in model.named_parameters():
        if 'register_network' not in n:
            p.requires_grad_(False)
    add_lora_by_name(model, permitted_parameters, lora_config=lora_config)

    if gaussian_noise:
        model.use_gaussian_noise()

    if learnable_noise:
        model.use_learnable_noise()

    model = lightning_fabric.setup(model)
    track_engine = TrackEngine(focal_length=12.0, device=device)
    feature_name = os.path.basename(image_path).split('.')[0]
    feature_data = get_tracked_results(image_path, track_engine, force_retrack=force_retrack)
    if feature_data is None:
        print(f'Finish inference, no face in input: {image_path}.')
        return
    # build driver data
    if os.path.isdir(driver_path):
        driver_name = os.path.basename(driver_path[:-1] if driver_path.endswith('/') else driver_path)
        driver_dataset = LoRAwRegistersDriverData(driver_path, feature_data, meta_cfg.DATASET.POINT_PLANE_SIZE)
        driver_dataloader = torch.utils.data.DataLoader(driver_dataset, batch_size=2, num_workers=6, shuffle=True, drop_last=True)
    else:
        print("Driver Path doesn't exist")

    driver_dataloader = lightning_fabric.setup_dataloaders(driver_dataloader)

    ### ----------- if you need to run multiview results ------------ ###
    # saving source image.
    dump_dir = os.path.join('render_results', feature_name)
    os.makedirs(dump_dir, exist_ok=True)
    # dump_path = os.path.join(dump_dir, f'{feature_name}.jpg')

    if threed_register:
        l_params = [
            {"params": list(get_lora_params(model)), 'lr': 1e-4, 'name':"whole_model"},
            {"params": model.register_network.parameters(), 'lr':1e-3, 'name': 'register_network'}
        ]
    elif learnable_noise:
        l_params = [
            {"params": list(get_lora_params(model)), 'lr': 1e-4, 'name':"whole_model"},
            {"params": model.learnable_embeddings, 'lr':1e-3, 'name':'learnable_embeddings'}
        ]
    else:
        l_params = [
            {"params": list(get_lora_params(model)), 'lr': 1e-4, 'name': "whole_model"},
        ]

    optimizer = Adam(l_params, lr=0.0, eps=1e-15)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.1,
        total_iters=num_iters,
    )
    driver_iterator = iter(driver_dataloader)
    for idx in tqdm(range(1, num_iters+1)):
        try:
            batch = next(driver_iterator)
        except StopIteration:
            driver_iterator = iter(driver_dataloader)
            batch = next(driver_iterator)
        render_results = model(batch)
        loss_metrics, show_metric = model.calc_metrics(render_results)
        wandb.log(loss_metrics)
        wandb.log(show_metric)
        if hasattr(model, "register_network"):
            if hasattr(model.register_network, "embeddings"):
                wandb.log({"register_embeddings_sum": torch.sum(model.register_network.embeddings.weight)})
        if threed_register_loss:
            if "output_sum" in render_results.keys():
                wandb.log({"output_sum": render_results['output_sum']})
        if idx % 10 == 0:
            source_images = wandb.Image(render_results['f_image'])
            target_images = wandb.Image(render_results['t_image'])
            gen_images = wandb.Image(render_results['gen_image'])
            sr_gen_images = wandb.Image(render_results['sr_gen_image'])
            wandb.log({'f_images': source_images, 't_image': target_images, 'gen_image': gen_images, 'sr_gen_images': sr_gen_images})
        loss = sum(loss_metrics.values())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        if idx%250==0:
            add_string = 'gaussian_noise_' if gaussian_noise else ''
            add_string = 'learnable_noise_' if learnable_noise else add_string
            add_string = 'threed_register_' if threed_register else add_string
            add_string = add_string + 'w_register_loss_' if threed_register_loss else add_string+''
            add_string = add_string + 'wo_feat_loss_' if remove_feat_loss else add_string
            add_string = add_string + 'wo_reg_loss_' if remove_reg_loss else add_string
            add_string = add_string + 'gaussian_noise_for_vertices_' if use_gaussian_noise_for_vertices else add_string
            add_string = add_string + 'remove_proc_1_' if remove_proc_1 else add_string
            add_string = add_string + f'lambda_reg_{lambda_reg}' if lambda_reg != 20.0 else add_string
            
            if threed_register:
                save_params={
                    'lora_weights': get_lora_state_dict(model),
                    'register_network': model.register_network.state_dict()
                }
            elif learnable_noise:
                save_params={
                    'lora_weights':get_lora_state_dict(model),
                    'learnable_embeddings': model.learnable_embeddings
                }
            else:
                save_params = {
                    'lora_weights': get_lora_state_dict(model),
                }
            torch.save(save_params, os.path.join(dump_dir, f'lora_all_{add_string}'+os.path.basename(driver_path)+'_'+str(idx)+'.pt'))
            torchvision.utils.save_image(render_results['gen_image'], os.path.join(dump_dir, f'lora_all_{add_string}'+os.path.basename(driver_path)+'_gen_image_'+str(idx)+'.png'))
            torchvision.utils.save_image(render_results['sr_gen_image'], os.path.join(dump_dir, f'lora_all_{add_string}'+os.path.basename(driver_path)+'_sr_gen_image_'+str(idx)+'.png'))
            torchvision.utils.save_image(render_results['t_image'], os.path.join(dump_dir, f'lora_all_{add_string}'+os.path.basename(driver_path)+'_tgt_image_'+str(idx)+'.png'))

    print(f'Finish finetuning: {dump_dir}.')


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
        if len(other_paths) <= 35:
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


if __name__ == '__main__':
    import warnings
    from tqdm.std import TqdmExperimentalWarning
    warnings.simplefilter("ignore", category=TqdmExperimentalWarning, lineno=0, append=False)
    # build args
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', '-i', required=True, type=str)
    parser.add_argument('--driver_path', '-d', required=True, type=str)
    parser.add_argument('--force_retrack', '-f', action='store_true')
    parser.add_argument('--resume_path', '-r', default='./assets/GAGAvatar.pt', type=str)
    parser.add_argument('--threed_register', action='store_true')
    parser.add_argument('--threed_register_loss', action='store_true')
    parser.add_argument('--gaussian_noise', action='store_true')
    parser.add_argument('--learnable_noise', action='store_true')
    parser.add_argument('--use_gaussian_noise_for_vertices', action='store_true')
    parser.add_argument('--remove_proc_1', action='store_true')
    parser.add_argument('--remove_feat_loss', help='remove the mse loss between target dino and reg module output', action='store_true')
    parser.add_argument('--remove_reg_loss', help='remove the regularization loss between reg module embeddings',
                        action='store_true')
    parser.add_argument('--lambda_reg', type=float, default=20.0)
    parser.add_argument('--iters', type=int, default=1000)
    args = parser.parse_args()
    # launch
    torch.set_float32_matmul_precision('high')
    lora_adaptation(args.image_path, args.driver_path, args.resume_path, force_retrack=args.force_retrack,
                    threed_register=args.threed_register, threed_register_loss=args.threed_register_loss,
                    gaussian_noise=args.gaussian_noise, learnable_noise=args.learnable_noise,
                    remove_feat_loss=args.remove_feat_loss, remove_reg_loss=args.remove_reg_loss, num_iters=args.iters,
                    use_gaussian_noise_for_vertices=args.use_gaussian_noise_for_vertices, remove_proc_1=args.remove_proc_1,
                    lambda_reg=args.lambda_reg)
