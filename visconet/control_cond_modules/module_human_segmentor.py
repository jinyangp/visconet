import os
import argparse
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, DeepLabV3_ResNet101_Weights
from visconet.control_cond_modules.util import resize_img_tensor

class HumanSegmentor(nn.Module):    
    
    def __init__(self,
                 model_name: str,
                 image_height: int = 768,
                 image_width: int = 768,
                 num_classes: int = 21):
        
        super().__init__()
        if model_name not in ("resnet_50", "resnet_101"):
            raise ValueError("Model name provided is invalid. Please use resnet_50 or resnet_101.")

        self.model_name = model_name

        self.image_height = image_height
        self.image_width = image_width

        self.num_classes = num_classes

        if model_name == "resnet_101":
            self.model = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT,
                                        num_classes=num_classes
                                        )
            self.transforms = DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1.transforms()

        else:
            self.model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT,
                                     num_classes=num_classes)
            self.transforms = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1.transforms()
    
        for params in self.model.parameters():
            params.requires_grad = False

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @torch.no_grad()
    def get_segmentation_masks(self,
                            img: Image.Image,
                            output_dir:str=None):

        transformed_img = torch.unsqueeze(self.transforms(img).to(self.device), 0)
        self.model.eval()
        model_output = self.model(transformed_img)  
        preds = model_output['out'][0].argmax(0)
        
        preds = preds.float()
        # NOTE: Model output shape != model input shape so we resize to ensure they are the same
        preds = resize_img_tensor(preds, self.image_height, self.image_width)
        preds = preds.squeeze(0)

        # class label of 15 for person in VOC and 1 for person in COCO
        # NOTE: 0 for background, 15 for human
        human_mask = (preds == 15).float()  # Assuming class label 15 represents 'person'
        background_mask = (preds == 0).float()  # Assuming class label 0 represents 'background'

        # Convert masks to images and save
        if output_dir:
            
            full_output_dir = os.path.join(os.getcwd(), output_dir)
            if not os.path.exists(full_output_dir):
                os.makedirs(full_output_dir)

            human_mask_image = Image.fromarray((human_mask.squeeze(0).numpy() * 255).astype('uint8'), mode='L')
            background_mask_image = Image.fromarray((background_mask.squeeze(0).numpy() * 255).astype('uint8'), mode='L')
            
            # Save the masks
            human_mask_image.save(os.path.join(full_output_dir, "human_mask.jpg"))
            background_mask_image.save(os.path.join(full_output_dir, "bg_mask.jpg"))
        
        # output shape: [original_image_height, original_image_width]
        return human_mask, background_mask

    @torch.no_grad()
    def forward(self,
                img: Image.Image,
                output_dir:str=None):
        
        img = img.resize((self.image_width, self.image_height))
    
        human_mask, background_mask = self.get_segmentation_masks(img,output_dir=output_dir)
        human_mask = human_mask.unsqueeze(0)
        # background_mask = background_mask.unsqueeze(0)

        img_tensor = torch.tensor(np.array(img)).to(self.device).permute(2,0,1)
        human_img_tensor = img_tensor * human_mask
        human_img_tensor = human_img_tensor.squeeze(0)

        # background_img_tensor = img_tensor * background_mask
        # background_img_tensor = background_img_tensor.squeeze(0)

        return human_img_tensor, human_mask.squeeze(0)