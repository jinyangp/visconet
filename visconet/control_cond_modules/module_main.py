import torch
import torch.nn as nn

from typing import Union, List
from PIL import Image

from ldm.util import instantiate_from_config

class LocalStyleProjector(nn.Module):

    def __init__(self, fashion_segmentor_config):

        '''
        This class contains the following blocks:
        1. FashionSegmentor: Takes in a fashion image and returns [N, 3, img_height, img_width]
        2. ImageEncoder: Takes in each cropped segmented image and returns [N, 1024] where each image is encoded to a dim of 1024
        3. TokenResampler: Takes in the embeddings of [N,1024] and returns the resampled embeddings of [N*num_queries, 1024]    
        '''
        super().__init__()
        self.fashion_segmentor = instantiate_from_config(fashion_segmentor_config)

    def forward(self,
                img: Image.Image):

        style_attrs = self.fashion_segmentor.get_style_attrs(img, "outputs/segmentation-test")
        return style_attrs