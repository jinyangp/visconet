import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from typing import Union, List
from einops import rearrange

from ldm.util import instantiate_from_config

class LocalStyleProjector(nn.Module):

    def __init__(self,
                 human_segmentor_config,
                 fashion_segmentor_config,
                 image_encoder_config,
                 resampler_config,
                 num_fashion_attrs:int=5,
                 uncond_guidance:bool=True,
                #  image_height:int=768, # 512 to align with image dimensions later on
                #  image_width:int=768
                 ):
        '''
        This class contains the following blocks:
        1. FashionSegmentor: Takes in a fashion image and returns [N, 3, img_height, img_width]
        2. ImageEncoder: Takes in each cropped segmented image and returns [N, 1024] where each image is encoded to a dim of 1024
        3. TokenResampler: Takes in the embeddings of [N,1024] and returns the resampled embeddings of [N*num_queries, 1024]    
        '''
        super().__init__()

        self.human_segmentor = instantiate_from_config(human_segmentor_config)
        self.fashion_segmentor = instantiate_from_config(fashion_segmentor_config)
        self.image_encoder = instantiate_from_config(image_encoder_config)
        self.resampler = instantiate_from_config(resampler_config)
        
        self.num_fashion_attrs = num_fashion_attrs
        self.uncond_guidance = uncond_guidance
        
        # self.image_height = image_height
        # self.image_width = image_width
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self,
                data_root_path: str,
                source_img_fp: str,
                seg_img_fp: str=None,
                output_dir: str=None):

        '''
        Pass a single image of shape [3,224,224] and get the resampled embeddings of it for each of the fashion attribute in it of shape [1,N*num_queries,1024]
        where N is the number of fashion attributes and num_queries is the number of queries per fashion attribute.
        
        Here the batch size is 1 since we process one image at a time only. While we pass in many source image as a batch during training, for each sample in the batch,
        we only have one source image that is passed through this module to process. Hence, LocalStyleProjector receives an input with a batch size of 1.
        
        Args:
            source_img_fp: str, filepath to the source image
            seg_img_fp: str, filepath to the segmentation image for the fashion attributes
            output_dir: str, output directory to the segmented fashion attribute

        Returns:
            dict of keys:
                human_mask, a tensor of shape [image_height, image_width]
                resampled_style_attrs_embed of shape [num_queries*num_fashion_attrs, emb_dim]
        '''

        source_img_fp = os.path.join(os.getcwd(), data_root_path, source_img_fp)
        source_img = Image.open(source_img_fp)

        # STEP: Segment into background and foreground
        human_img_tensor, human_mask = self.human_segmentor(source_img, output_dir=output_dir)
        print(f'Segmented human image: {human_img_tensor.shape}')
        print(f'Segmented human mask: {human_mask.shape}')

        # STEP: Segment the source image for fashion attributes
        # output_shape: [num_detected_attrs, 3, 224, 224] where 0 <= num_detected_attrs <= num_fashion_attrs
        # if we already have the segmentation ap for fasihon attributes,
        if seg_img_fp and os.path.exists(os.path.join(os.getcwd(),data_root_path, seg_img_fp)):
            seg_img_fp = os.path.join(os.getcwd(),data_root_path, seg_img_fp)
            style_attrs = self.fashion_segmentor(human_img_tensor, seg_img_fp=seg_img_fp, output_dir=output_dir)
        # else, if we need to manually segment,
        else:
            style_attrs = self.fashion_segmentor(human_img_tensor, output_dir=output_dir)

        # output_shape: [num_fashion_attrs, 3, 224, 224]
        num_attrs = style_attrs.shape[0]
        num_null_attrs = self.num_fashion_attrs - num_attrs
        # if we need to pad with 0 matrices
        if num_null_attrs > 0:
            _, ch, height, width = style_attrs.shape
            null_attrs = torch.zeros((num_null_attrs, ch, height, width), dtype=style_attrs.dtype)
            style_attrs = torch.cat([style_attrs, null_attrs], dim=0)
        # if we need to remove some fashion attributes
        elif num_null_attrs < 0:
            style_attrs = style_attrs[:self.num_fashion_attrs,:,:,:]
        print(f'Segmented style attrs shape: {style_attrs.shape}')

        # STEP: Encode the fashion attributes
        # output shape: [num_fashion_attrs, 257, 1024] if using CLIP embeddor
        # output shape: [num_fashion_attrs, 257, 768] if using DINO embeddor
        # NOTE: If uncond_guidance is used,
        if self.uncond_guidance:
            uncond_attrs = torch.zeros_like(style_attrs, dtype=style_attrs.dtype)
            uncond_attrs_embed = self.image_encoder(uncond_attrs)

        style_attrs_embed = self.image_encoder(style_attrs)
        print(f'Encoded style attrs shape: {style_attrs_embed.shape}')
        print(f'Unconditional attrs embed shape: {uncond_attrs_embed.shape}')

        # STEP: Resample the embeddings
        # output shape: [num_fashion_attrs*num_queries, 1024]
        # [num_fashion_attrs, num_queries, 1024]
        resampled_style_attrs_embed = self.resampler(style_attrs_embed)
        # [num_fashion_attrs, num_queries, 1024] -> [num_fashion_attrs*num_queries, 1024]
        resampled_style_attrs_embed = rearrange(resampled_style_attrs_embed, 'b n d -> (b n) d')

        resampled_uncond_attrs_embed = self.resampler(uncond_attrs_embed)
        resampled_uncond_attrs_embed = rearrange(resampled_uncond_attrs_embed, 'b n d -> (b n) d')

        print(f'Resampled Encoded style attrs shape: {resampled_style_attrs_embed.shape}')
        print(f'Resampled Unconditional attrs embed shape: {resampled_uncond_attrs_embed.shape}')

        # TODO: Return a dictionary instead so we can return the mask as well
        return resampled_style_attrs_embed