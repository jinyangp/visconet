import os
import numpy as np
import torch
import torchvision
from PIL import Image
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.distributed import rank_zero_only

from visconet.control_cond_modules.util import resize_img_tensor

class StylesLogger(Callback):

    # TODO: Change batch_frequency to 1 for now for debugging purposes, originally 100
    def __init__(self, batch_frequency=2000, folder_name="", disabled=False,
                 image_height=224, image_width=224):
        super().__init__()
        self.batch_freq = batch_frequency
        self.folder_name = os.path.join("image_log", folder_name)
        self.disabled = False
        self.grid_image_height = image_height
        self.grid_image_width = image_width

    # this decorator ensures that this function runs only on the rank:0 GPU
    @rank_zero_only
    def log_styles(self, pl_module, batch, batch_idx, global_step, current_epoch, split="train"):

        save_dir = pl_module.logger.save_dir
        root = os.path.join(save_dir, self.folder_name, split)
        device = pl_module.device
        num_fashion_attrs = pl_module.control_cond_model.num_fashion_attrs

        # STEP: Prepare images to make grid
        src_img_pils = batch["src_img_pil"]
        seg_img_pils = batch["seg_img_pil"]
        imgs = []

        src_pils = zip(src_img_pils, seg_img_pils)
        
        for src_img, seg_img in src_pils:
            human_img_tensor, human_mask = pl_module.control_cond_model.human_segmentor(src_img)
            if seg_img:
                style_attrs = pl_module.control_cond_model.fashion_segmentor(human_img_tensor, seg_img=seg_img)
            else:
                style_attrs = pl_module.control_cond_model.fashion_segmentor(human_img_tensor)
            
            num_attrs = style_attrs.shape[0]
            num_null_attrs = num_fashion_attrs - num_attrs
            # if we need to pad with 0 matrices
            if num_null_attrs > 0:
                _, ch, height, width = style_attrs.shape
                null_attrs = torch.zeros((num_null_attrs, ch, height, width), dtype=style_attrs.dtype).to(device)
                style_attrs = torch.cat([style_attrs, null_attrs], dim=0)   
            # if we need to remove some fashion attributes  
            elif num_null_attrs < 0:
                style_attrs = style_attrs[:num_fashion_attrs,:,:,:]

            human_img_tensor = human_img_tensor.unsqueeze(0)
            src_img = torch.tensor(np.array(src_img)).to(device).permute(2,0,1).unsqueeze(0)
            
            # STEP: Resize human image tensor and source image to match style attributes so that
            # we can make image grid later
            human_img_tensor = resize_img_tensor(human_img_tensor, self.grid_image_height, self.grid_image_width)
            src_img = resize_img_tensor(src_img, self.grid_image_height, self.grid_image_width)

            # [7,C,H,W]
            img = torch.cat((src_img, human_img_tensor, style_attrs), dim=0).to(device)
            imgs.append(img)
            # TO ENSURE: ALL IMAGE IN (C,H,W) with same width and height

        # [BS,7,C,H,W]
        imgs = torch.stack(imgs,dim=0).to(device)
        nrow = imgs.shape[1]
        imgs = imgs.view(-1, imgs.shape[2], imgs.shape[3], imgs.shape[4])

        # STEP: Make image grid
        grid = torchvision.utils.make_grid(imgs, nrow=nrow)

        # STEP: Convert grid to numpy array
        grid_np = grid.permute(1, 2, 0).detach().cpu().numpy()
        grid_np = grid_np.astype(np.uint8)

        # STEP: Save image
        filename = "styles_gs-{:06}_e-{:06}_b-{:06}.png".format(global_step, current_epoch, batch_idx)
        path = os.path.join(root, filename)
        os.makedirs(os.path.split(path)[0], exist_ok=True)
        Image.fromarray(grid_np).save(path) # currently, grid is of shape (1,1,5,224)?

    def check_frequency(self, check_idx):
        return check_idx % self.batch_freq == 0
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled and self.check_frequency(batch_idx):
            self.log_styles(pl_module, batch, batch_idx,
                            pl_module.global_step, pl_module.current_epoch,
                            split="train")
        
    def on_val_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled and self.check_frequency(batch_idx):
            self.log_styles(pl_module, batch, batch_idx,
                            pl_module.global_step, pl_module.current_epoch,
                            split="val")