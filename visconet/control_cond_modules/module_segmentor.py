import os
import cv2
import torch
import torch.nn as nn
import numpy as np

from typing import List
from PIL import Image
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation

class FashionSegmentor(nn.Module):

    def __init__(self,
                 seg_processor,
                 seg_model,
                 ignore_labels: List[str] = ["Belt", "Scarf", "Bag", "Left-leg",
                                             "Right-leg", "Left-arm", "Right-arm"],
                 ):
        
        # processor = SegformerImageProcessor.from_pretrained("sayeed99/segformer_b3_clothes")
        # model = AutoModelForSemanticSegmentation.from_pretrained("sayeed99/segformer_b3_clothes")
        
        super().__init__()
        self.model = AutoModelForSemanticSegmentation.from_pretrained(seg_model)    
        
        img_size = self.model.config.image_size
        self.processor = SegformerImageProcessor.from_pretrained(seg_processor,
                                                                 do_resize = True,
                                                                 size={
                                                                     "height": 224,
                                                                     "width": 224
                                                                 })
        
        self.img_size = img_size
        self.ignore_labels = ignore_labels

        original_id2label = self.model.config.id2label
        self._id2label = {k:v for k,v in original_id2label.items() if v not in self.ignore_labels}
        self._ids = [k for k in self._id2label.keys()]


    def _reverse_normalise(self, normalised_tensor, mask):
        '''
        Reverse normalization to get original pixel values.

        Args:
        - tensor (Tensor): Normalized tensor.
        - mean (list): Mean values for each channel.
        - std (list): Standard deviation values for each channel.

        Returns:
        - original_tensor (Tensor): Original tensor.
        '''



        mean = torch.tensor(self.processor.image_mean).view(1, 3, 1, 1)
        std = torch.tensor(self.processor.image_std).view(1, 3, 1, 1)
        # Reverse normalization only for non-masked pixels
        original_tensor = torch.where(mask.unsqueeze(0) == 0, 
                                    torch.zeros_like(normalised_tensor), 
                                    (normalised_tensor * std) + mean)
        original_tensor = torch.clamp(original_tensor, 0, 1)
        return original_tensor
       

    def postprocess(self, org_img_tensor, seg_img_tensor, output_dir):

        ''' 
        To receive the segmentation mask and get a masked image for each attribute detected

        Args:
            org_img_tensor: tensor, of original image
            seg_img_tensor: tensor, segmentation mask with mask values being class labels
        Returns:
            tensor of shape [N, 3, 224, 224] where N is the number of fashion attributes detected in the segmentation
            mask
        '''

        # (num_attrs, 3, org_img_height, org_img_width)
        masked_imgs = []            
        present_ids = torch.unique(seg_img_tensor)

        # STEP: Iterate through list of ids we are trying to find
        for _id in present_ids:

            # NOTE: We can skip _ids that are not in our target list
            if _id.item() not in self._ids:
                continue

            # STEP: Get mask for this particular key
            mask = (seg_img_tensor == _id).to(torch.uint8)
            # STEP: Apply mask for this particular key
            masked_img_tensor= org_img_tensor * mask.unsqueeze(0)
            masked_imgs.append(masked_img_tensor)

            if output_dir:
                full_output_dir = os.path.join(os.getcwd(), output_dir)
                if not os.path.exists(full_output_dir):
                    os.makedirs(full_output_dir)
                
                masked_img_cv = self._reverse_normalise(masked_img_tensor, mask)
                masked_img_cv = torch.squeeze(masked_img_cv, 0)
                masked_img_cv = masked_img_cv.permute(1,2,0).numpy()    
                masked_img_cv = (masked_img_cv * 255).astype(np.uint8)
                              
                label_name = self._id2label[_id.item()]
                filename = f'{label_name}-{_id.item()}.png'
                full_fp = os.path.join(full_output_dir, filename)

                masked_img_pil = Image.fromarray(masked_img_cv)
                masked_img_pil.save(full_fp)

        # STEP: Finally, return the list of tensor imgs as a tensor
        return torch.cat(masked_imgs, dim=0)


    def get_style_attrs(self,
                        img: Image.Image,
                        output_dir: str):
        '''
        To pass in an image and return the segmented mask
        Args:
            img: Image.Image, of image to be segmented
        Returns:
            style_attrs: tensor of shape [N, 3, org_img_height, org_img_width] where N is the number
            of fashion attributes
        '''

        img = img.resize((self.img_size, self.img_size))
        org_img = self.processor(images=img, return_tensors="pt") 
        org_img_tensor = org_img["pixel_values"]
        out = self.model(**org_img)
        logits = out.logits

        upsampled_logits = nn.functional.interpolate(logits,
                                                     size=img.size[::-1],
                                                     mode="bilinear",
                                                     align_corners=False
                                                     )
        
        seg_img_tensor = upsampled_logits.argmax(dim=1)[0]        
        style_attrs = self.postprocess(org_img_tensor, seg_img_tensor, output_dir)
        return style_attrs
    

        