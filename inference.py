import os
import torch
import argparse
import numpy as np
import einops
import random

from datetime import datetime
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from pathlib import Path
from huggingface_hub import snapshot_download
from PIL import Image

import config
from annotator.util import pad_image, resize_image, HWC3
from ldm.util import instantiate_from_config
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler

from einops import rearrange
from annotator.openpose.get_pose_hf import get_openpose_annotations
from torchvision import transforms as T
from torchvision.utils import make_grid

def log_sample(seed, results, prompt, skeleton_image,  mask_image, control_scales, *viscon_images):
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    APP_FILES_PATH = Path('./app_files')
    LOG_PATH = APP_FILES_PATH/'logs'
    
    log_dir = LOG_PATH/time_str
    os.makedirs(str(log_dir), exist_ok=True)

    # save result
    concat = np.hstack((skeleton_image, *results))
    Image.fromarray(skeleton_image).save(str(log_dir/'skeleton.jpg'))   
    Image.fromarray(mask_image).save(str(log_dir/'mask.png'))
    for i, result in enumerate(results):
        Image.fromarray(result).save(str(log_dir/f'result_{i}.jpg'))

    # save text
    with open(str(log_dir/'info.txt'),'w') as f:
        f.write(f'prompt: {prompt} \n')
        f.write(f'seed: {seed}\n')
        control_str = [str(x) for x in control_scales]
        f.write(','.join(control_str) + '\n')
    # save vison images
    for style_name, style_image in zip(style_names, viscon_images):
        if style_image is not None:
            style_image.save(str(log_dir/f'{style_name}.jpg'))
'''
if log_samples:
    log_sample(seed, results, prompt, detected_map, mask_image, control_scales, *viscon_images)
'''

H = 512
W = 512

def valid_float_list(value):
    try:
        values = list(map(float, value.split(',')))
        if len(values) != 13:
            raise ValueError("List must contain exactly 13 elements.")
        if not all(0.0 <= v <= 1.0 for v in values):
            raise ValueError("All values must be within the range [0.0, 1.0].")
        return values
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e))

if __name__ == "__main__":

    '''
    NOTE: Sample command
    srun -p rtx3090_slab -n 1 --job-name=test --gres=gpu:1 --kill-on-bad-exit=1 python3 -u inference.py data/datasets/deepfashion/imgs/MEN/Denim/id_00000080/01_7_additional.jpg \
    data/datasets/deepfashion/imgs/MEN/Denim/id_00000089/45_7_additional.jpg --output_dir=logs/inference-test  --gpu 0 --config=./configs/visconet_v15_pair.yaml --ckpt=./logs/270125-expt1/last.ckpt \
    --prompt='a person, plain studio background' --n_prompt='deformed body parts, blurry, noisy background, low definition' --num_samples=3 --ddim_steps=100 --cfg_scale=10.
    '''

    parser = argparse.ArgumentParser()

    # Get the current timestamp
    current_timestamp = datetime.now()
    # Format the timestamp in DDMMYY-HH:MM:SS format
    formatted_timestamp = current_timestamp.strftime("%d%m%y-%H:%M:%S")

    # Input & Output filepaths
    parser.add_argument("src_image_fp", type=str, help="Relative filepath to source image from current directory")
    parser.add_argument("tgt_image_fp", type=str, help="Relative filepath to target image from current directory")
    parser.add_argument("--output_dir", type=str, default=os.path.join(os.getcwd(), "logs", formatted_timestamp), help="Output directory path")
    parser.add_argument("--seg_image_fp", type=str, default=None, help="Relative filepath to fashion seg image from current directory")
    # GPU and model configurations
    parser.add_argument('--gpu', type=int, default=0, help='GPU id')
    parser.add_argument('--config', type=str, default='./configs/visconet_v1.yaml', help="relative filepath to model config file")
    parser.add_argument('--ckpt', type=str, default='./models/visconet_v1.pth', help="relative filepath to model checkpoint file")
    parser.add_argument('--prompt', type=str, default="", help="text prompt about object to condition image generation")
    parser.add_argument("--n_prompt", type=str, default="", help="negative text prompt about image style to pull image generation away")
    parser.add_argument("--num_samples", type=int, default=3, help="number of samples to generate")
    parser.add_argument("--ddim_steps", type=int, default=50, help="number of inference steps")
    parser.add_argument("--cfg_scale", type=float, default=7.5, help="value of control free guidance scale")
    parser.add_argument("--seed", type=int, default=42, help="seed to base generation on")
    parser.add_argument("--eta", type=float, default=0., help="eta value that controls how much noise is added to image at each reverse step in diffusion process")
    
    parser.add_argument("--controlnet_scales", type=valid_float_list, default=[1.0]*13, help="Comma-separated list of 13 float values between 0.0 and 1.0")
    parser.add_argument("--log_controlnet_bias_scales_grid", action="store_true", help="Whether to log a grid for different controlnet and bias scales.")
    args = parser.parse_args()

    # STEP: Get the file path to the source and target image
    src_img = Image.open(os.path.join(os.getcwd(), args.src_image_fp))
    tgt_img = Image.open(os.path.join(os.getcwd(), args.tgt_image_fp))
    seg_img = Image.open(os.path.join(os.getcwd(), args.seg_image_fp)) if args.seg_image_fp else None

    if args.output_dir:
        os.makedirs(os.path.join(os.getcwd(), args.output_dir), exist_ok=True)
    
    # STEP: Initialise model and load it in to device
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    config_file = args.config
    model_ckpt = args.ckpt
    proj_config = OmegaConf.load(config_file)
    model = create_model(config_file).cpu()  
    model.load_state_dict(load_state_dict(model_ckpt, location=device))
 
    model = model.to(device)   
    ddim_sampler = DDIMSampler(model)

    # NOTE: Image preprocessing transformations for input
    image_shape = [512,512]
    image_tform = T.Compose([
        T.Resize(image_shape),
        T.ToTensor(),
        T.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
    
    skeleton_tform = T.Compose([
        T.Resize(image_shape),
        T.ToTensor()
    ])

    num_samples = args.num_samples
    # STEP: Get control scales
    control_scales = args.controlnet_scales
    model.control_scales = control_scales

    # STEP: Get the mask and the fashion attribute encodings from target image (from get_input function of model)
    dct = model.control_cond_model(src_img, tgt_img, seg_img=seg_img)
    style_attrs_embeds, human_mask = dct["style_attr_embeds"].to(device), dct["human_mask"].to(device)
    style_attrs_embeds = style_attrs_embeds.unsqueeze(0).repeat(num_samples,1,1)
    human_mask = human_mask.unsqueeze(0).repeat(num_samples,1,1) # TODO: Create one variable for the repeated and not repeated version

    # STEP: Settle text prompts
    model.cond_stage_model.device = device # Load embeddor for text prompt
    c_text = model.get_learned_conditioning([args.prompt] * num_samples)
    uncond_text = model.get_learned_conditioning([args.n_prompt] * num_samples)
    
    # STEP: Get OpenPose image
    target_pose_pil = get_openpose_annotations(tgt_img)
    pose_img = skeleton_tform(target_pose_pil)
    pose_img = torch.stack([pose_img for _ in range(num_samples)], dim=0).to(device)

    if config.save_memory:
        model.low_vram_shift(is_diffusing=True)

    cond = {
        "c_concat": [pose_img], # [3,3,512,512]
        "c_crossattn": [style_attrs_embeds], # [3,num_queries,query_dim]
        "c_text": [c_text],
        "c_concat_mask": [human_mask]
    }
    un_cond = {
        "c_concat": [pose_img],
        "c_crossattn": [torch.zeros_like(style_attrs_embeds).to(device)],
        "c_text": [uncond_text],
        "c_concat_mask": [torch.zeros_like(human_mask).to(device)]
    }

    # STEP: Get source image embeddings
    if model.use_bias:
        src_img_latent = src_img
        if model.bias_mask_only:
            src_img_latent, _ = model.control_cond_model.human_segmentor(src_img)
        encoder_posterior = model.encode_first_stage(src_img_latent.unsqueeze(0))
        src_img_latent = model.get_first_stage_encoding(encoder_posterior).detach()
        src_img_latent = src_img_latent.repeat(num_samples,1,1,1)

        cond['c_src'] = [src_img_latent]
        un_cond['c_src'] = [torch.zeros_like(src_img_latent)]

    latent_shape = (4, H//8, W//8)
    samples, _ = ddim_sampler.sample(args.ddim_steps, num_samples, latent_shape,
                                     cond, verbose=False, eta=args.eta,
                                     unconditional_guidance_scale=args.cfg_scale,
                                     unconditional_conditioning=un_cond)

    if config.save_memory:
        model.low_vram_shift(is_diffusing=False)

    x_samples = model.decode_first_stage(samples)
    x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

    results = [x_samples[i] for i in range(num_samples)]

    control_net_scales = [i*0.2 for i in range(0,6)] # [i*0.2 for i in range(1,6)]
    bias_scales = [i*0.2 for i in range(0,6)] # [i*0.2 for i in range(1,6)] 
    if args.log_controlnet_bias_scales_grid:
        assert control_net_scales and bias_scales, "ucg_values and ddim_steps values must be provided to plot grid."

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        # STEP: Generate images
        grid_images = []
        for cs in control_net_scales:
            for bs in bias_scales:
                    
                control_scales = [cs]*13
                model.control_scales = control_scales   
                model.bias_scale = bs
                
                samples, _ = ddim_sampler.sample(args.ddim_steps, num_samples, latent_shape,
                                    cond, verbose=False, eta=args.eta,
                                    unconditional_guidance_scale=args.cfg_scale,
                                    unconditional_conditioning=un_cond)
                
                x_samples = model.decode_first_stage(samples)
                grid_images.append(x_samples)

            if config.save_memory:
                model.low_vram_shift(is_diffusing=False)

        grid_images = torch.stack(grid_images, dim=0) # shape [NC,B,C,H,W] where NC is the number of combinations of parameters to test for
        grid_images = grid_images.permute(1,0,2,3,4) # shape [B, NC, C, H, W]
        grid_images = torch.clamp(grid_images.detach().cpu() * 0.5 + 0.5, 0., 1.)
        grid_images = torch.clamp(grid_images*255., min=0., max=255.) # to convert grid to range of [0,255]
        img_grid = make_grid(grid_images.squeeze(0), nrow=len(control_net_scales), padding=5, pad_value=1.0)
        img_grid_np = img_grid.permute(1,2,0).detach().cpu().numpy()
        img_grid_np = img_grid_np.astype(np.uint8)

        if args.output_dir:
            filename='controlnet_bias_scales_grid.png'
            path = os.path.join(args.output_dir, filename)
            Image.fromarray(img_grid_np).save(path)

    if args.output_dir:
        os.makedirs(os.path.join(os.getcwd(), args.output_dir), exist_ok=True)
        for idx,res in enumerate(results):
            res_pil = Image.fromarray(res)
            res_pil.save(os.path.join(args.output_dir, f"image_{idx}.png"))