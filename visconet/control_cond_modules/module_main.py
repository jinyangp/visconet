import torch
import torch.nn as nn

from typing import Union, List
from PIL import Image

from ldm.util import instantiate_from_config

class LocalStyleProjector(nn.Module):

    def __init__(self, fashion_segmentor_config, image_encoder_config):

        '''
        This class contains the following blocks:
        1. FashionSegmentor: Takes in a fashion image and returns [N, 3, img_height, img_width]
        2. ImageEncoder: Takes in each cropped segmented image and returns [N, 1024] where each image is encoded to a dim of 1024
        3. TokenResampler: Takes in the embeddings of [N,1024] and returns the resampled embeddings of [N*num_queries, 1024]    
        '''
        super().__init__()
        self.fashion_segmentor = instantiate_from_config(fashion_segmentor_config)
        self.image_encoder = instantiate_from_config(image_encoder_config)

    def forward(self,
                img: Image.Image):

        # TODO: Replace with the actual source image later on
        # output_shape: [num_attrs, 3, 224, 224]
        style_attrs = self.fashion_segmentor.get_style_attrs(img, "outputs/segmentation-test")
        # output shape: [1, num_attrs, 257, 1024] if using CLIP embeddor
        # output shape: [1, num_attrs, 257, 768] if using DINO embeddor
        style_attrs_embed = self.image_encoder(style_attrs).unsqueeze(0)
        return style_attrs_embed