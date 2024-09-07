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

def encode_style_images(style_images):
    style_embeddings = []

    for style_name, style_image in zip(style_names, style_images):
        if style_image == None:
            style_image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
            
        #style_image = style_image.resize((224,224))            
        style_image = style_encoder.preprocess(style_image).to(device)
        style_emb = style_encoder.postprocess(style_encoder(style_image)[0])
        style_embeddings.append(style_emb)

    styles = torch.tensor(np.array(style_embeddings)).squeeze(-2).unsqueeze(0).float().to(device)
    return styles


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


# NOTE: This is the function that does the generation
def process(prompt, a_prompt, n_prompt, num_samples,
            ddim_steps, scale, seed, eta, mask_image, pose_image,  
            control_scales, log_samples, *viscon_images):
    '''
    prompt and a_prompt: To be used as positive prompt in generation 
    n_prompt: To be used as negative prompt in generation
    ddim_steps: int, number of inference steps
    scale: float, CFG scale
    seed: int, to control randomisation
    eta: float, controls the amount of noise added during the
    reverse process of the diffusion model
    Used as such: 
        cond = {"c_concat": [control], 
            "c_crossattn": [style_emb.repeat(new_style_shape)],
            "c_text": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)],
            'c_concat_mask': [mask.repeat(num_samples, 1, 1, 1)]}
    num_samples: number of images to generate for
    mask_image: mask of the person (foreground) excluding background
    pose_image: RGB pose image of the desired pose
    c0-c12: control scales to use for the middle block and the output blocks
    viscon_images: style images, cropped out fashion attributes
    '''
    
    with torch.no_grad():
        # control_scales = [c12, c11, c10, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0]
        mask = torch.tensor(mask_image.mean(-1)/255.,dtype=torch.float) #(512,512), [0,1]
        mask = mask.unsqueeze(0).to(device) # (1, 512, 512)
        # NOTE: Get embeddings of style
        style_emb = encode_style_images(viscon_images)

        # fix me
        # NOTE: Get poses
        detected_map = HWC3(pose_image)
        #detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)
        H, W, C = detected_map.shape
        control = torch.from_numpy(detected_map.copy()).float().to(device) / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)
        new_style_shape = [num_samples] + [1] * (len(style_emb.shape)-1)

        # for CFG
        cond = {"c_concat": [control], 
                "c_crossattn": [style_emb.repeat(new_style_shape)],
                "c_text": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)],
                'c_concat_mask': [mask.repeat(num_samples, 1, 1, 1)]}
        un_cond = {"c_concat": [control], 
                   "c_crossattn": [torch.zeros_like(style_emb).repeat(new_style_shape)],
                   "c_text":[model.get_learned_conditioning([n_prompt] * num_samples)],
                   'c_concat_mask': [torch.zeros_like(mask).repeat(num_samples, 1, 1, 1)]}
        
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = control_scales

        samples, _ = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]

    if log_samples:
        log_sample(seed, results, prompt, detected_map, mask_image, control_scales, *viscon_images)
    return results


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # TODO: Need to handle images
    parser.add_argument("mask_image_fp", type=str, help="filepath to binary mask of subject/background")
    parser.add_argument("pose_image_fp", type=str, help="filepath to RGB pose image")

    parser.add_argument('--gpu', type=int, default=0, help='GPU id')
    parser.add_argument('--config', type=str, default='./configs/visconet_v1.yaml', help="relative filepath to model config file")
    parser.add_argument('--ckpt', type=str, default='./models/visconet_v1.pth', help="relative filepath to model checkpoint file")

    parser.add_argument('--prompt', type=str, default="", help="text prompt about object to condition image generation")
    parser.add_argument("--a_prompt", type=str, default="", help="additional text prompt about image style to condition image generation")
    parser.add_argument("--n_prompt", type=str, default="", help="negative text prompt about image style to pull image generation away")
    parser.add_argument("--num_samples", type=int, default=3, help="number of samples to generate")
    parser.add_argument("--ddim_steps", type=int, default=50, help="number of inference steps")
    parser.add_argument("--cfg_scale", type=float, default=7.5, help="value of control free guidance scale")
    parser.add_argument("--seed", type=int, default=42, help="seed to base generation on")
    parser.add_argument("--eta", type=float, default=0., help="eta value that controls how much noise is added to image at each reverse step in diffusion process")
    parser.add_argument("--control_scale_type", type=str, default="Default", help="key value to the control scales (c0-c12) type configuration")
    parser.add_argument("--log_samples", action="store_true", help="boolean whether to save generated samples")
     
    parser.add_argument("--viscon_face_img", type=str, default=None, help="Visual conditioning of the face using a face image.")
    parser.add_argument("--viscon_hair_img", type=str, default=None, help="Visual conditioning of the hair using a hair image.")
    parser.add_argument("--viscon_headwear_img", type=str, default=None, help="Visual conditioning of the hair using a headwear image.")
    parser.add_argument("--viscon_top_img", type=str, default=None, help="Visual conditioning of the hair using a top image.")
    parser.add_argument("--viscon_outer_img", type=str, default=None, help="Visual conditioning of the hair using a outer image.")
    parser.add_argument("--viscon_bottom_img", type=str, default=None, help="Visual conditioning of the hair using a bottom image.")
    parser.add_argument("--viscon_shoes_img", type=str, default=None, help="Visual conditioning of the hair using a shoes image.")
    parser.add_argument("--viscon_accessory_img", type=str, default=None, help="Visual conditioning of the hair using accesories image.")
    
    args = parser.parse_args()

    global device
    global segmentor
    global apply_openpose
    global style_encoder
    global model
    global ddim_sampler    
    global dataset
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    config_file = args.config
    model_ckpt = args.ckpt

    proj_config = OmegaConf.load(config_file)
    style_names = proj_config.dataset.train.params.style_names
    
    print(style_names) # check what style names are we expected to give

    # Get Visconet model weights
    HF_REPO = 'soonyau/visconet'
    if not os.path.exists(model_ckpt):
        snapshot_download(repo_id=HF_REPO, local_dir='./models',
                        allow_patterns=os.path.basename(model_ckpt))

    # Get style encoder
    style_encoder = instantiate_from_config(proj_config.model.style_embedding_config).to(device)
    model = create_model(config_file).cpu()    
    model.load_state_dict(load_state_dict(model_ckpt, location=device))

    model = model.to(device)

    # Load embeddor for text prompt
    model.cond_stage_model.device = device
    ddim_sampler = DDIMSampler(model)

    mask_img = np.array(Image.open(os.path.join(os.getcwd(), args.mask_image_fp)))
    pose_img = np.array(Image.open(os.path.join(os.getcwd(), args.pose_image_fp)))

    DEFAULT_CONTROL_SCALE = 1.0
    SCALE_CONFIG = {
        'Default': [DEFAULT_CONTROL_SCALE]*13, 
        'DeepFakes':[1.0, 1.0, 1.0,
                    1.0, 1.0, 1.0,
                    0.5, 0.5, 0.5,
                    0.0, 0.0, 0.0, 0.0,],    
        'Faithful':[1,1,1,
                    1,1,1,
                    1,1,0.5,
                    0.5,0.5,0,0],
        'Painting':[0.0,0.0,0.0,
                    0.5,0.5,0.5,
                    0.5,0.5,0.5,
                    0.5,0,0,0],
        'Pose':    [0.0,0.0,0.0,
                    0.0,0.0,0.0,
                    0.0,0.0,0.5,
                    0.0,0.0,0,0],
        'Texture Transfer':  [1.0,1.0,1.0,
                    1.0,1.0,1.0,
                    0.5,0.0,0.5,
                    0.0,0.0,0,0]
        }
    
    control_scale_type = args.control_scale_type
    control_scales = SCALE_CONFIG[control_scale_type]

    viscon_images_fp = [args.viscon_face_img, args.viscon_hair_img, args.viscon_headwear_img, args.viscon_top_img,
                     args.viscon_outer_img, args.viscon_bottom_img, args.viscon_shoes_img, args.viscon_accessory_img]
    viscon_images = []
    for fp in viscon_images_fp:
        if fp:
            viscon_images.append(Image.open(os.path.join(os.getcwd(), fp)))
        else:
            viscon_images.append(None)

    process(args.prompt, args.a_prompt, args.n_prompt, args.num_samples, args.ddim_steps,
            args.cfg_scale, args.seed, args.eta, mask_img, pose_img, control_scales,
            args.log_samples, *viscon_images)