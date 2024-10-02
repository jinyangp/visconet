import os
import cv2
import torch
import torch.nn as nn
import numpy as np

from typing import List, Tuple
from PIL import Image
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation

class FashionSegmentor(nn.Module):

    def __init__(self,
                 seg_processor: str,
                 seg_model: str,
                 output_shape: Tuple[int, int] = (224,224),
                 ignore_labels: List[str] = ["Belt", "Scarf", "Bag", "Left-leg",
                                             "Right-leg", "Left-arm", "Right-arm"],
                 ):
        
        # processor = SegformerImageProcessor.from_pretrained("sayeed99/segformer_b3_clothes")
        # model = AutoModelForSemanticSegmentation.from_pretrained("sayeed99/segformer_b3_clothes")
        
        super().__init__()
        self.model = AutoModelForSemanticSegmentation.from_pretrained(seg_model)    
        self.model = self.model.eval()
        
        img_size = self.model.config.image_size
        self.processor = SegformerImageProcessor.from_pretrained(seg_processor,
                                                                 do_resize = True,
                                                                 size={
                                                                     "height": img_size,
                                                                     "width": img_size
                                                                 })
        
        self.img_size = img_size
        self.output_shape = output_shape
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

            # STEP: Convert masked image tensor back to numpy
            masked_img_org_vals = self._reverse_normalise(masked_img_tensor, mask)
            masked_img_org_vals = torch.squeeze(masked_img_org_vals, 0)
            masked_img_org_vals_np = masked_img_org_vals.permute(1,2,0).numpy()
            masked_img_org_vals_np = (masked_img_org_vals_np * 255).astype(np.uint8)

            # STEP: Crop, Resize and Centralise attributes - Processed np array
            masked_img_org_vals_np = self._crop_and_recentre(masked_img_org_vals_np,
                                                             new_img_size=self.output_shape)

            # STEP: If output_dir provided, save the processed np array as image
            if output_dir:
                    
                full_output_dir = os.path.join(os.getcwd(), output_dir)
                if not os.path.exists(full_output_dir):
                    os.makedirs(full_output_dir)
                    
                label_name = self._id2label[_id.item()]
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
    

        