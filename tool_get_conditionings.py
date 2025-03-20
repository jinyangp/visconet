import os
import torch
import argparse
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from datetime import datetime

from cldm.model import create_model
from annotator.openpose.get_pose_hf import get_openpose_annotations

# root path to all deepfashion images
IMG_ROOT_DIR = os.path.join(os.getcwd(), "data", "datasets", "deepfashion", "imgs")
FILE_EXT = "png"
# initialise model
config_file = os.path.join(os.getcwd(), "configs", "visconet_v7_pair.yaml")
model = create_model(config_file).cpu()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

def convert_fname(x):
    a, b = os.path.split(x)
    i = b.rfind('_')
    x = a + '/' +b[:i] + b[i+1:]
    return 'fashion'+x.split('.jpg')[0].replace('id_','id').replace('/','')

def get_human_mask(img_fp, output_dir=None):
    
    img_full_fp = os.path.join(IMG_ROOT_DIR, img_fp)
    img_pil = Image.open(img_full_fp)

    _, human_mask = model.control_cond_model.human_segmentor(img_pil)

    if output_dir:
        
        # Make directory
        output_dir = os.path.join(os.getcwd(), "logs", output_dir)
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        # Save image
        # (C,H,W) -> (H,W,C)
        human_mask_np = human_mask.permute(1,2,0).detach().cpu().numpy()
        human_mask_np = (human_mask_np*255).astype(np.uint8)
        human_mask_np = np.squeeze(human_mask_np) 
        human_mask_pil = Image.fromarray(human_mask_np)
        human_mask_pil.save(os.path.join(output_dir, f'mask_{convert_fname(img_fp)}.{FILE_EXT}'))

    return human_mask

def get_skeletal_map(img_fp, output_dir=None):
    
    img_full_fp = os.path.join(IMG_ROOT_DIR, img_fp)
    img_pil = Image.open(img_full_fp)

    pose_pil = get_openpose_annotations(img_pil)

    if output_dir:

        # Make directory
        output_dir = os.path.join(os.getcwd(), "logs", output_dir)
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        pose_pil.save(os.path.join(output_dir, f'skeletal_{convert_fname(img_fp)}.{FILE_EXT}'))

    return pose_pil

if __name__ == "__main__":

    current_timestamp = datetime.now()
    # Format the timestamp in DDMMYY-HH:MM:SS format
    formatted_timestamp = current_timestamp.strftime("%d%m%y-%H:%M:%S")

    parser = argparse.ArgumentParser(description='Description of your program.')
    parser.add_argument("img_fp", type=str, help="File path to image to be analysed.")
    parser.add_argument("--output_dir", type=str, default=f"logs/{formatted_timestamp}", help="Output directory to store results in.")
    parser.add_argument("--get_mask", action="store_true", help="Flag to determine if get human mask.")
    parser.add_argument("--get_skeletal_map", action="store_true", help="Flag to determine if get human pose.")
    parser.add_argument("--get_fashion_attributes", action="store_true", help="Flag to determine if get fashion attributes.")
    args = parser.parse_args()

    if args.get_mask:
        get_human_mask(args.img_fp, output_dir=args.output_dir)
    
    if args.get_skeletal_map:
        get_skeletal_map(args.img_fp, output_dir=args.output_dir)
    
    if args.get_fashion_attributes:
        get_fashion_attributes(args.img_fp, output_dir=args.output_dir)