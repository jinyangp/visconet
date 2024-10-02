import torch
import torch.nn as nn
import numpy as np

from abc import ABC, abstractmethod
from transformers import AutoModel, AutoImageProcessor, CLIPVisionModelWithProjection, CLIPProcessor

class ImageEncoder(ABC, nn.Module):

    def __init__(self,
                 encoder_processor_name: str,
                 encoder_model_name: str,
                 ):
        
        super().__init__()
        self.encoder_processor_name = encoder_processor_name
        self.encoder_model_name = encoder_model_name

    @abstractmethod
    def preprocess(self, images):
        '''
        To preprocess the image to a shape compatible with the encoder_model using the encoder_processor
        '''
        pass

    @abstractmethod
    def postprocess(self, embeddings):
        '''
        To postprocess the image embeddings from encoder_model into a shape that can be passed on down the pipeline
        '''
        pass


class CLIPImageEncoder(ImageEncoder):

    def __init__(self,
                 encoder_type,
                 encoder_processor_name,
                 encoder_model_name):
        
        super().__init__(encoder_processor_name,
                         encoder_model_name)
        self.encoder_type = encoder_type
        self.encoder_processor = CLIPProcessor.from_pretrained(self.encoder_processor_name)
        self.encoder_model = CLIPVisionModelWithProjection.from_pretrained(self.encoder_model_name)

        self.encoder_model = self.encoder_model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def preprocess(self, images):
        '''
        Args:
            images: tensor of shape [N, 3, 224, 224], a batch of N images in its pixel values

        Returns:
            tensor, tensor of shape [N,3,224,224], a batch of N images in processed pixel values ready for model
        '''
        x = torch.tensor(self.encoder_processor.image_processor(images).pixel_values)
        return x

    def postprocess(self, embeddings):
        '''
        Args:
            embeddings: tensor of shape (N,257,1024), embeddings of N images with sequence length 257 and embed dim of 1024 
        '''
        return embeddings.detach()
    
    def forward(self, images):
        
        style_images = self.preprocess(images)
        ret = self.encoder_model(style_images)
        style_embeds = self.postprocess(ret[1])
        return style_embeds
    

class DINOImageEncoder(ImageEncoder):
    
    def __init__(self,
                 encoder_type,
                 encoder_processor_name,
                 encoder_model_name):
        super().__init__(encoder_processor_name,
                         encoder_model_name)
        self.encoder_type = encoder_type
        self.encoder_processor = AutoImageProcessor.from_pretrained(encoder_processor_name)
        self.encoder_model = AutoModel.from_pretrained(encoder_model_name)
        
        self.encoder_model = self.encoder_model.eval()
        for param in self.parameters():
            param.requires_grad = False
    
    # def _tensor_to_pil(self, images):
    #     from torchvision.transforms import ToPILImage
    #     return [ToPILImage()(img) for img in images]

    def preprocess(self, images):
        # pil_images = self._tensor_to_pil(images)
        # return self.encoder_processor(images=pil_images, return_tensors="pt")
        return self.encoder_processor(images=images, return_tensors="pt")
    
    def postprocess(self, embeddings):
        return embeddings.detach()
    
    def forward(self, images):
        style_images = self.preprocess(images)
        style_embeds = self.postprocess(self.encoder_model(**style_images)[0])
        return style_embeds
