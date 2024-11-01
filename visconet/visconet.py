import os
import einops
import torch
import numpy as np
from torchvision import transforms as T
from pathlib import Path
from PIL import Image
from einops import rearrange, repeat
from torchvision.utils import make_grid
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import log_txt_as_img, exists, instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

from visconet.control_cond_modules.util import resize_img_tensor

class ViscoNetLDM(LatentDiffusion):

    def __init__(self, control_stage_config, control_key, only_mid_control, control_cond_config, 
                 control_crossattn_key, mask_key=None, enable_mask=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.control_model = instantiate_from_config(control_stage_config)
        self.control_key = control_key # image pose prompt - for openpose
        self.control_crossattn_key = control_crossattn_key # image pose prompt - for fashion attribute styles
        self.only_mid_control = only_mid_control
        self.control_scales = [1.0] * 13
        self.enable_mask = enable_mask
        self.mask_enables = [1 if enable_mask else 0] * 13
        self.mask_key = mask_key
        self.ddim_sampler = DDIMSampler(self)
        # new
        self.control_cond_model = instantiate_from_config(control_cond_config)
    
    '''
    # NOTE: get_input() and apply_model() are used behind the scenes in the training_step which is a necessary step needed to be implemented to use Pytorch Lightning
    # .fit() function
    '''

    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        # STEP: Get latents of image and text embeddings
        # NOTE: first_stage_key is "jpg" and it refers to the target image so 
        # NOTE: 
        x, c_text = super().get_input(batch, self.first_stage_key, *args, **kwargs)

        # STEP: Get pose
        control = batch[self.control_key]
        if bs is not None:
            control = control[:bs]
        control = control.to(self.device)
        control = einops.rearrange(control, 'b h w c -> b c h w')
        control = control.to(memory_format=torch.contiguous_format).float()
        
        # STEP: Start to format a dictionary to give to model
        ret_dict = dict(c_text=[c_text], c_concat=[control])

        def format_input(key):
            val = batch[key]
            if bs is not None:
                val = val[:bs]
            val = val.to(memory_format=torch.contiguous_format).float()
            val = val.to(self.device)
            return val
        
        # STEP: Use the src_img key from our batch to get the style attrs and human_mask
        src_img_pils = batch["src_img_pil"]
        seg_img_pils = batch["seg_img_pil"]
        target_img_pils = batch["target_img_pil"]
        if bs is not None:
            src_img_pils = src_img_pils[:bs]
            seg_img_pils = seg_img_pils[:bs]
            target_img_pils = target_img_pils[:bs]

        # STEP: Run source image pil through our localstyleprojector module
        src_pils = zip(src_img_pils, seg_img_pils, target_img_pils)
        style_attrs = []
        human_masks = []
        target_img_pils = []
        for src_img, seg_img, target_img in src_pils:
            dct = self.control_cond_model(src_img, seg_img, target_img)
            style_attr_embeds = dct["style_attr_embeds"]
            human_mask = dct["human_mask"]

            style_attrs.append(style_attr_embeds)
            human_masks.append(human_mask)

        if self.control_crossattn_key:
            ret_dict["c_crossattn"] = [torch.stack(style_attrs,dim=0)]
        
        if self.mask_key:
            ret_dict["c_concat_mask"] = [torch.stack(human_masks,dim=0)]

        # NOTE: Old way
        # -------
        # if self.control_crossattn_key:
        #     ret_dict['c_crossattn']=[format_input(self.control_crossattn_key)]

        # if self.mask_key:
        #     ret_dict['c_concat_mask']=[format_input(self.mask_key)]
        # -------

        return x, ret_dict

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model
        # c_concat : skeleton [batch, 3, 512, 512] -> NOTE: the openpose
        # c_crossattn : text [batch, 77, 1024] -> NOTE: the style attributes

        cond_txt = torch.cat(cond['c_text'], 1) # remove list
        cond_cross = torch.cat(cond['c_crossattn'], 1) 
        cond_mask = torch.cat(cond['c_concat_mask'], 1)
        cond_concat = torch.cat(cond['c_concat'], 1) # the openpose pose

        if cond['c_concat'] is None:
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
        else:
            control = self.control_model(x=x_noisy,
                                         hint=cond_concat,
                                         timesteps=t, 
                                         context=cond_cross)

            def mask_control(c, mask_enable):
                # shapes --> cond_mask: [2,512,512], resized_mask shape: [2,64,64], c aka controlnet output shape: [2, 320, 64, 64] 
                if mask_enable:
                    resized_mask = T.Resize(list(c.shape[-2:]), T.InterpolationMode.NEAREST)(cond_mask)
                    resized_mask = resized_mask.unsqueeze(1)
                    return c * resized_mask
                else:
                    return c
            
            # Get the list o controls by applying the mask level-wise to each level's output    
            control = [mask_control(c, mask_enable) * scale for c, scale, mask_enable in zip(control, self.control_scales, self.mask_enables)]
            # NOTE: Run it through the UNET model forward function with the signature
            # def forward(self, x, timesteps=None, context=None, y=None,**kwargs):
            # NOTE: We are using the ControlledUnetModel class in the config file which overrides original forward method o 
            # UNET to use control and only_mid_control arguments
            
            # STEP: If IP-Adapter is being used here, we concatenate them along the same dimension and chunk them for processing
            # concatenate by torch.cat((cond_text,cond_img), dim=1)
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)

        return eps

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):

        return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def log_images(self, batch, N=2, n_row=2, sample=False, ddim_steps=20, ddim_eta=0.0, 
                   plot_denoise_rows=False, plot_diffusion_rows=False, unconditional_guidance_scale=12.0,**kwargs):
        use_ddim = ddim_steps is not None

        log = dict()
        
        # NOTE: The image being fed into the VAE to get the latents is the target image
        # NOTE: The pose comes from the target image as well
        # NOTE: But the style comes from the source image

        # NOTE: z holds the latents of the source image
        z, c = self.get_input(batch, self.first_stage_key, bs=N)
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)

        # c_concat: the openpose image
        # c_text: the text prompt
        # c_concat_mask: the human mask to apply
        # c_crossattn: the style attributes
        c_cat, c_text, mask = c["c_concat"][0][:N], c["c_text"][0][:N], c["c_concat_mask"][0][:N]
        c = c["c_crossattn"][0][:N]

        reconstructed = self.decode_first_stage(z)[:N]

        log["control"] = c_cat * 2.0 - 1.0
        log["conditioning"] = log_txt_as_img((64, 64), batch[self.cond_stage_key], size=16)

        # NOTE: if we want to see rows of diffusion outputs
        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, sunglasses, hat'

        cond = {"c_concat": [c_cat], # openpose pose
                "c_crossattn": [c], # the style attrs
                "c_text": [c_text], 
                'c_concat_mask': [mask]}

        un_cond = {"c_concat": [c_cat], 
                "c_crossattn": [torch.zeros_like(c)],
                "c_text":[self.get_learned_conditioning([n_prompt] * N)],
                'c_concat_mask': [torch.zeros_like(mask)] }
        
        # NOTE: if we just want to compare the sampled output with the source image
        if sample:
            # get denoise row
            samples, z_denoise_row = self.sample_log(cond=cond,
                                                     batch_size=N, ddim=use_ddim,
                                                     ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)
            #log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            
            #uc_cross = self.get_unconditional_conditioning(N)
            uc_cat = c_cat  # torch.zeros_like(c_cat)
            #uc_full = {"c_concat": [uc_cat], "c_text": [uc_cross], "c_crossattn": [torch.zeros_like(c)], 'c_concat_mask':[mask]}
            samples_cfg, _ = self.sample_log(cond=cond,
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=un_cond,
                                             )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            # log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg
            
            # NOTE: samples refers to the reconstructed image generated with guidance
            log["samples"] = x_samples_cfg

            # NOTE: reconstructed shape: [min(bs,N),3,512,512], x_sample_cfg: [min(bs,N),3,512,512]
            
            # Initialize an empty list to store transformed tensors
            src_img_pils = batch["src_img_pil"][:N]
            src_img_tensors = []

            # Process each PIL image
            for pil_img in src_img_pils:
                tensor_img = T.ToTensor()(pil_img)  # Convert to tensor in [0, 1]
                tensor_img = tensor_img * 2 - 1  # Rescale to [-1, 1]
                src_img_tensors.append(tensor_img)
            # Stack the list into a single tensor with shape [N, C, H, W] if needed
            src_img_tensors = torch.cat([torch.stack(src_img_tensors,dim=0)], 1)# expect [min(bs,N), 3, pil_img_height, pil_img_width]
            
            model_output_height, model_output_width = x_samples_cfg.shape[-2], x_samples_cfg.shape[-1]
            src_imgs = resize_img_tensor(src_img_tensors, model_output_height, model_output_width).to(self.device) # expect [min(bs,N), 3, model_output_height, model_output_width]
             
            # Initialize an empty list to store transformed tensors
            # target_img_pils = batch["target_img_pil"][:N]
            # target_img_tensors = []

            # Process each PIL image
            # for pil_img in target_img_pils:
            #     tensor_img = T.ToTensor()(pil_img)  # Convert to tensor in [0, 1]
            #     tensor_img = tensor_img * 2 - 1  # Rescale to [-1, 1]
            #     target_img_tensors.append(tensor_img)
            # # Stack the list into a single tensor with shape [N, C, H, W] if needed
            # target_img_tensors = torch.cat([torch.stack(target_img_tensors,dim=0)], 1)# expect [min(bs,N), 3, pil_img_height, pil_img_width]
            
            # model_output_height, model_output_width = x_samples_cfg.shape[-2], x_samples_cfg.shape[-1]
            # target_imgs = resize_img_tensor(target_img_tensors, model_output_height, model_output_width).to(self.device) # expect [min(bs,N), 3, model_output_height, model_output_width]
             
            # NOTE: in order, concat shows the source images where the styles are taken from, the reconstructed target images and the generated images
            log['concat'] = torch.cat((src_imgs, reconstructed, x_samples_cfg), dim=-2)

        return log

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):

        b, c, h, w = cond["c_concat"][0].shape
        shape = (self.channels, h // 8, w // 8)
        samples, intermediates = self.ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        return samples, intermediates

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        # create folder
        f_ext = 'png' 
        sample_root = Path(self.logger.save_dir)/'samples'
        gt_root = Path(self.logger.save_dir)/'gt'
        #mask_root = Path(self.logger.save_dir)/'mask'
        src_root = Path(self.logger.save_dir)/'src'
        concat_root = Path(self.logger.save_dir)/'concat'

        for root_name in [sample_root, gt_root, src_root, concat_root]:
            os.makedirs(str(root_name), exist_ok=True)        
        # inference
        images = self.log_images(batch, N=len(batch), ddim_steps=24, 
                                 unconditional_guidance_scale=14.0, sample=True)
        
        images['samples'] = torch.clamp(images['samples'].detach().cpu() * 0.5 + 0.5, 0., 1.1)
        images['samples']/=(torch.max(torch.abs(images['samples'])))

        for k in ['src_img', 'jpg']:
            batch[k] = (rearrange(batch[k],'b h w c -> b c h w') + 1.0) / 2.0

        # save ground truth, source, mask
        # save samples
        for  sample, fname, src_image, gt in \
            zip(images['samples'], batch['fname'], batch['src_img'], batch['jpg']):

            #resized_mask = T.Resize(list(sample.shape[-2:]), T.InterpolationMode.NEAREST)(mask).to(sample.device)
            #sample *= resized_mask

            #neg_mask = (~resized_mask.bool())*torch.tensor(1.0, dtype=torch.float32).to(sample.device)
            #sample += neg_mask * T.Resize(sample.shape[-2:])(bg).to(sample.device)
            
            sample = T.CenterCrop(size=(512, 352))(sample)
            gt = T.CenterCrop(size=(512, 352))(gt)
            src_image = T.CenterCrop(size=(512, 352))(src_image)
            concat = T.Resize([256, 528])(torch.cat([src_image.detach().cpu(),
                                                     gt.detach().cpu(),
                                                     sample.detach().cpu()], 2))

            T.ToPILImage()(concat).save(concat_root/f'{fname}.{f_ext}')
        
            T.ToPILImage()(sample).save(sample_root/f'{fname}.{f_ext}')
            T.ToPILImage()(src_image).save(src_root/f'{fname}.{f_ext}')
            T.ToPILImage()(gt).save(gt_root/f'{fname}.{f_ext}')
            #T.ToPILImage()(mask).save(mask_root/f'{fname}.{f_ext}')
    
    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.control_model.parameters())
        params += list(self.control_cond_model.parameters())
        if not self.sd_locked:
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.to(self.device)
            self.control_model = self.control_model.to(self.device)
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.control_model = self.control_model.cpu()
            self.first_stage_model = self.first_stage_model.to(self.device)
            self.cond_stage_model = self.cond_stage_model.to(self.device)

