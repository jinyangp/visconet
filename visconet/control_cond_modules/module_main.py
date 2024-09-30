import torch
import torch.nn as nn

from typing import Union, List
from PIL import Image

from ldm.util import instantiate_from_config

class LocalStyleProjector(nn.Module):

    def __init__(self,
                 fashion_segmentor_config,
                 image_encoder_config,
                 num_fashion_attrs:int=8,
                 uncond_guidance:bool=True
                 ):
        '''
        This class contains the following blocks:
        1. FashionSegmentor: Takes in a fashion image and returns [N, 3, img_height, img_width]
        2. ImageEncoder: Takes in each cropped segmented image and returns [N, 1024] where each image is encoded to a dim of 1024
        3. TokenResampler: Takes in the embeddings of [N,1024] and returns the resampled embeddings of [N*num_queries, 1024]    
        '''
        super().__init__()
        self.fashion_segmentor = instantiate_from_config(fashion_segmentor_config)
        self.image_encoder = instantiate_from_config(image_encoder_config)
        self.num_fashion_attrs = num_fashion_attrs
        self.uncond_guidance = uncond_guidance
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self,
                img:Image.Image):

        # TODO: Replace with the actual source image later on
        
        # STEP: Segment the source image for fashion attributes
        # output_shape: [num_detected_attrs, 3, 224, 224] where 0 <= num_detected_attrs <= num_fashion_attrs
        style_attrs = self.fashion_segmentor.get_style_attrs(img, "outputs/segmentation-test")

        # output_shape: [num_fashion_attrs, 3, 224, 224]
        num_attrs = style_attrs.shape[0]
        num_null_attrs = self.num_fashion_attrs - num_attrs
        if num_null_attrs:
            _, ch, height, width = style_attrs.shape
            null_attrs = torch.zeros_like((num_null_attrs, ch, height, width), dtype=style_attrs.dtype)
            style_attrs = torch.cat([style_attrs, null_attrs], dim=0)


        # output shape: [1, num_fashion_attrs, 257, 1024] if using CLIP embeddor
        # output shape: [1, num_fashion_attrs, 257, 768] if using DINO embeddor
        # NOTE: Here the batch size is 1 since we process one image at a time only. Each sample in the batch only has one
        # source image so our LocalStyleProjector receives input with a batch size of 1
        # STEP: Encode the fashion attributes 
        # NOTE: If uncond_guidance is used,
        if self.uncond_guidance:
            uncond_attrs = torch.zeros_like(style_attrs, dtype=style_attrs.dtype)
            uncond_attrs_embed = self.image_encoder(uncond_attrs).unsqueeze(0)

        style_attrs_embed = self.image_encoder(style_attrs).unsqueeze(0)

        # TODO: Based on image encoder used, pass in the embedding dim to the Resampler of IPAdapter
        return style_attrs_embed