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
    print("running on device:", device)
    model = model.to(device)
    # to run with an already segmented image
    # outs = model("app_files/default_images",
    #              "fashionWOMENBlouses_Shirtsid0000047902_4full.jpg",
    #              seg_img_fp="WOMEN-Blouses_Shirts-id_00000479-02_4_full_segm.png",
    #              output_dir="outputs/segmentation-test-3"
    #              )
    # to run without an already segmented image
    outs = model("app_files/default_images",
                 "fashionWOMENBlouses_Shirtsid0000047902_4full.jpg",
                 seg_img_fp="WOMEN-Blouses_Shirts-id_00000479-02_4_full_segm.png",
                 output_dir="outputs/segmentation-test-with-segmap"
                 )
    
    style_attrs = outs["style_attr_embeds"]
    human_mask = outs["human_mask"]
    
    print(style_attrs.shape, human_mask.shape)