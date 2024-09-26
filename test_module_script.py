'''
TODO: THIS IS A TEMPORARY SCRIPT USED TO TEST THE MODULE
'''

import os
import torch
from omegaconf import OmegaConf
from PIL import Image

from cldm.model import create_model

if __name__ == "__main__":

    config_file = os.path.join(os.getcwd(), "configs", "localstyleprojector_v1.yaml")
    model = create_model(config_file).cpu()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    img_fp = os.path.join(os.getcwd(), "app_files", "default_images", "ref.png")
    img = Image.open(img_fp)
    outs = model(img)

    print(outs.shape)