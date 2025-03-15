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

if __name__ == "__main__":

    '''
    NOTE: Sample command
    srun -p rtx3090_slab -n 1 --job-name=test --gres=gpu:1 --kill-on-bad-exit=1 python3 -u inference.py data/datasets/deepfashion/imgs/MEN/Denim/id_00000080/01_7_additional.jpg \
    data/datasets/deepfashion/imgs/MEN/Denim/id_00000089/45_7_additional.jpg --output_dir=logs/inference-test  --gpu 0 --config=./configs/visconet_v15_pair.yaml --ckpt=./logs/270125-expt1/last.ckpt \
    --prompt='a person, plain studio background' --n_prompt='deformed body parts, blurry, noisy background, low definition' --num_samples=3 --ddim_steps=100 --cfg_scale=10.
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='GPU id')
    parser.add_argument('--config', type=str, default='./configs/visconet_v1.yaml', help="relative filepath to model config file")
    args = parser.parse_args()
    
    # STEP: Initialise model and load it in to device
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    config_file = args.config
    proj_config = OmegaConf.load(config_file)
    model = create_model(config_file).cpu()  
    model = model.to(device)   

    model.get_num_parameters()