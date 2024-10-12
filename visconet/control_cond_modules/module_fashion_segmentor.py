import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from typing import List, Tuple
from PIL import Image
from torchvision import transforms
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation

from visconet.control_cond_modules.util import resize_img_tensor

class FashionSegmentor(nn.Module):

    def __init__(self,
                 seg_processor: str,
                 seg_model: str,
                 valid_threshold: float = 0.002,
                 output_shape: Tuple[int, int] = (224,224),
                 ignore_labels: List[str] = ["Belt", "Scarf", "Bag", "Left-leg",
                                             "Right-leg", "Left-arm", "Right-arm",
                                             "Background", "Sunglasses"],
                 target_labels: List[str] = ["Face", "Hair", "Pants", "Upper-clothes",
                                            "Left-shoe", "Right-shoe"],
                 default_seg_map_id2label = {
                                           1: "top",
                                           5: "pants",
                                           11: "footwear",
                                           13: "hair",
                                           14: "face"
                                           }
                 ):
        
        '''
        Args:
            seg_processor: str, name of segmentation processor from HF
            seg_model: str, name of segmentation model from HF
            valid_threshold: float, percentage of style attribute pixels in an image to be considered valid
            output_shape: Tuple[int,int], output shape of each style attribute
            ignore_labels: List[str], list of labels to ignore in the segmented output from HF model
            target_labels: List[str], list of target labels we want from the segmented output from HF model
            default_seg_map_labels: Dict, key-value pairs of segmented fashion attribute from segmentation map provided by DeepFashion multimodal dataset, if provided
        '''
        
        # processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
        # model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
        
        super().__init__()
        self.model = AutoModelForSemanticSegmentation.from_pretrained(seg_model)    
        for param in self.model.parameters():
            param.requires_grad = False
        self.model = self.model.eval()
        
        img_size = self.model.config.image_size
        self.processor = SegformerImageProcessor.from_pretrained(seg_processor,
                                                                 do_resize=False
                                                                 )
        self.valid_threshold = valid_threshold

        self.img_size = img_size # 224,224
        self.output_shape = output_shape # 224,224
        self.model_ignore_labels = ignore_labels
        self.model_target_labels = target_labels

        original_id2label = self.model.config.id2label
        self.model_id2label = {k:v for k,v in original_id2label.items() if v not in self.model_ignore_labels}
        self.model_ids = [k for k in self.model_id2label.keys()]
        self.model_target_ids = [k for k,v in self.model_id2label.items() if v in self.model_target_labels] # [2, 6, 9, 10, 11]

        self.default_seg_map_id2label = default_seg_map_id2label

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

        mean = torch.tensor(self.processor.image_mean).view(1, 3, 1, 1).to(normalised_tensor.device)
        std = torch.tensor(self.processor.image_std).view(1, 3, 1, 1).to(normalised_tensor.device)
        
        # Reverse normalization only for masked areas
        original_tensor = (normalised_tensor * std) + mean
        # Mask out irrelevant parts by applying the mask
        original_tensor = original_tensor * mask.unsqueeze(0).float()
        # Clip the values to the [0, 1] range to avoid artifacts
        original_tensor = torch.clamp(original_tensor, 0, 1)

        return original_tensor
    

    def _crop_and_recentre(self,
                           img_np,
                           padding = 5,
                           new_img_size = (224,224)):
        '''
        Crops out the attribute and recentres it by resizing it to the specified output shape.

        Args:
            - img_np (np.ndarray): Numpy array of image
            - padding (int): Number of extra pixels to crop around border of fashion attribute
            - new_img_size (tuple[int]): Tuple of desired output shape 

        Returns:
            - np.ndarray, of cropped and recentred fashion attribute.
        '''

        indices = np.where(img_np != 0)
        indices = np.array([(indices[0][i], indices[1][i]) for i in range(len(indices[0]))])

        y_indices = indices[:,0]
        x_indices = indices[:,1]

        y_min, y_max = min(y_indices), max(y_indices)
        x_min, x_max = min(x_indices), max(x_indices)

        org_img_height, org_img_width = img_np.shape[0], img_np.shape[1]

        y_min = y_min - padding if y_min >= padding else y_min
        y_max = y_max + padding if y_max + padding < org_img_height else y_max
        x_min = x_min - padding if x_min >= padding else x_min
        x_max = x_max + padding if x_max + padding < org_img_width else x_max

        # STEP: Crop
        cropped_img_np = img_np[y_min:y_max, x_min:x_max,:]
        
        # STEP: Resize
        cropped_img_pil = Image.fromarray(cropped_img_np) 
        resized_img_pil = cropped_img_pil.resize(new_img_size)

        return np.array(resized_img_pil)            

    def is_attr_valid(self,
                     masked_img_np
                     ):
        
        '''
        Given an image with the mask applied, return the image as valid if the percentage of non-zero pixel is more than the threshold
        
        Args:
            masked_img_np: np.ndarray, numpy array of masked image

        Returns:
            bool, whether the fashion attribute is valid for use
        '''
        
        mask_pixels = np.all(masked_img_np == 0, axis=0)
        num_mask_pixels = np.sum(mask_pixels)
        total_pixels = masked_img_np.shape[0]*masked_img_np.shape[1]
        mask_pixels_pct = num_mask_pixels/total_pixels

        return mask_pixels_pct < self.valid_threshold

    def postprocess(self,
                    org_img_tensor,
                    seg_img_tensor,
                    target_label_dict,
                    use_seg_model: bool=False,
                    output_dir: str=None):

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
        target_ids =  [k for k in target_label_dict.keys()]

        # STEP: Iterate through list of ids we are trying to find
        for _id in present_ids:

            # NOTE: We can skip _ids that are not in our target list
            if _id.item() not in target_ids:
                continue

            print(f"Found id: {_id.item()} referring to class {target_label_dict[_id.item()]}") # TODO: Delete later, for debuggging

            # STEP: Get mask for this particular key
            mask = (seg_img_tensor == _id).to(torch.uint8)

            # STEP: Apply mask for this particular key
            masked_img_tensor = org_img_tensor * mask.unsqueeze(0)

            # STEP: Convert masked image tensor back to numpy
            if use_seg_model:
                masked_img_org_vals = self._reverse_normalise(masked_img_tensor, mask)
                masked_img_org_vals = torch.squeeze(masked_img_org_vals, 0)
                masked_img_org_vals_np = masked_img_org_vals.permute(1,2,0).numpy()
                masked_img_org_vals_np = (masked_img_org_vals_np * 255).astype(np.uint8)

            else:
                masked_img_org_vals_np = masked_img_tensor.permute(1,2,0).numpy().astype(np.uint8)

            # STEP: Crop, Resize and Centralise attributes - Processed np array
            masked_img_org_vals_np = self._crop_and_recentre(masked_img_org_vals_np,
                                                            new_img_size=self.output_shape)
            
            # STEP: Only check for validity if we are using the HF segmentation model and add this attribute to output array if the number of pixels is above valid threshold
            if not use_seg_model or self.is_attr_valid(masked_img_org_vals_np):

                print(f"Found and valid id: {_id.item()} referring to class {target_label_dict[_id.item()]}") # TODO: Delete later, for debuggging
                
                # STEP: If output_dir provided, save the processed np array as image
                if output_dir:
                        
                    full_output_dir = os.path.join(os.getcwd(), output_dir)
                    if not os.path.exists(full_output_dir):
                        os.makedirs(full_output_dir)
                        
                    label_name = target_label_dict[_id.item()]
                    filename = f'{label_name}-{_id.item()}.png'
                    full_fp = os.path.join(full_output_dir, filename)

                    masked_img_pil = Image.fromarray(masked_img_org_vals_np)
                    masked_img_pil.save(full_fp)

                # STEP: Append the processed np array converted to a tensor to res array
                masked_img_org_vals_tensor = torch.from_numpy(masked_img_org_vals_np)
                masked_img_org_vals_tensor = masked_img_org_vals_tensor.permute(2,0,1)
                masked_img_org_vals_tensor = masked_img_org_vals_tensor.unsqueeze(0)

                masked_imgs.append(masked_img_org_vals_tensor)

        # STEP: Concate all tensors in the res arr and return
        return torch.cat(masked_imgs, dim=0)

    def forward(self,
                img_tensor,
                seg_img_fp:str=None,
                output_dir:str=None):
        
        org_height, org_width = img_tensor.shape[1], img_tensor.shape[2]

        # expects seg_mask and image to be of the following shapes
        # torch.Size([768, 768]) torch.Size([3, 768, 768])
        
        # STEP: If the default segmented fashion attributes are provided, we can use the segmentation output from there
        if seg_img_fp:
            seg_img_tensor = torch.tensor(np.array(Image.open(seg_img_fp)))
            seg_img_tensor = resize_img_tensor(seg_img_tensor, org_height, org_width)
            
            # resizing using interpolation requires us to unsqueeze till 4 dimensions so we need to squeeze it back to 2 dimensions
            while len(seg_img_tensor.shape) > 2:
                seg_img_tensor = seg_img_tensor.squeeze(0)
            # seg_img_tensor = seg_img_tensor.to(torch.int)

            style_attrs = self.postprocess(img_tensor,
                                           seg_img_tensor,
                                           self.default_seg_map_id2label,
                                           output_dir=output_dir)    
        # STEP: Else if not provided, we have to segment the image with the model from HF
        else:
            org_img = self.processor(images=img_tensor, return_tensors="pt") 
            org_img_tensor = org_img["pixel_values"] # [1,3,768,768]
            out = self.model(**org_img)
            logits = out.logits

            org_height, org_width = img_tensor.shape[1], img_tensor.shape[2]
            upsampled_logits = resize_img_tensor(logits, org_height, org_width)
            
            seg_img_tensor = upsampled_logits.argmax(dim=1)[0]

            style_attrs = self.postprocess(org_img_tensor,
                                           seg_img_tensor,
                                           self.model_id2label,
                                           use_seg_model=True,
                                           output_dir=output_dir)
        
        return style_attrs        
        
