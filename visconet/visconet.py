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
                 control_crossattn_key, src_encoder_config=None, mask_key=None, enable_mask=True, p_cg=None, use_bias=False,
                 bias_mask_only=False, use_lora=False, lora_apply_mask_only=False ,*args, **kwargs):
        super().__init__(*args, **kwargs, lora_apply_mask_only=lora_apply_mask_only)
        self.control_model = instantiate_from_config(control_stage_config)
        self.control_key = control_key # image pose prompt - for openpose
        self.control_crossattn_key = control_crossattn_key # image pose prompt - for fashion attribute styles
        self.only_mid_control = only_mid_control
        self.control_scales = [1.0] * 13
        self.bias_scale = 1.0
        self.enable_mask = enable_mask
        self.mask_enables = [1 if enable_mask else 0] * 13
        self.mask_key = mask_key
        self.ddim_sampler = DDIMSampler(self)

        # NOTE: NEW
        self.control_cond_model = instantiate_from_config(control_cond_config)
        self.p_cg = p_cg
        if self.p_cg: # NOTE: If ucg is to be used, assign a value of 0.05 in config YAML file
            self.cg_prng = np.random.RandomState()

        self.use_bias = use_bias
        self.bias_mask_only = bias_mask_only
        if self.use_bias:
            self.src_encoder = instantiate_from_config(src_encoder_config)

        self.use_lora = use_lora

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
        # src_img_pils = batch["src_img_pil"]
        seg_img_pils = batch["seg_img_pil"]
        src_img_pils = batch["src_img_pil"]
        target_img_pils = batch['target_img_pil']
        if bs is not None:
            # src_img_pils = src_img_pils[:bs]
            seg_img_pils = seg_img_pils[:bs]
            src_img_pils = src_img_pils[:bs]
            target_img_pils = target_img_pils[:bs]

        src_pils = zip(seg_img_pils, src_img_pils, target_img_pils)
        style_attrs = []
        human_masks = []
        for seg_img, style_img, target_img in src_pils:
            dct = self.control_cond_model(style_img, target_img, seg_img=seg_img)
            style_attr_embeds = dct["style_attr_embeds"]
            human_mask = dct["human_mask"]

            style_attrs.append(style_attr_embeds)
            human_masks.append(human_mask)

        if self.control_crossattn_key:
            ret_dict["c_crossattn"] = [torch.stack(style_attrs,dim=0)]
        
        if self.mask_key:
            ret_dict["c_concat_mask"] = [torch.stack(human_masks,dim=0)]

        if self.use_bias:
            if self.bias_mask_only:
                src_img_masks = []
                for src_img in src_img_pils:
                    src_img_mask, _ = self.control_cond_model.human_segmentor(src_img)
                    encoder_posterior = self.encode_first_stage(src_img_mask.unsqueeze(0))
                    src_x = self.get_first_stage_encoding(encoder_posterior).detach()
                    src_img_masks.append(src_x.squeeze(0))
                ret_dict['c_src'] = [torch.stack(src_img_masks, dim=0)]
            else:
                src_x, _ = super().get_input(batch, "src_img", *args, **kwargs)
                ret_dict["c_src"] = [src_x]
  
        # NOTE: Old way
        # -------
        # if self.control_crossattn_key:
        #     ret_dict['c_crossattn']=[format_input(self.control_crossattn_key)]

        # if self.mask_key:
        #     ret_dict['c_concat_mask']=[format_input(self.mask_key)]
        # -------

        return x, ret_dict

    def training_step(self, batch, batch_idx):

        # STEP: Get inputs; each key would become a list of tensors
        # z shape: [BS, C, H, W]
        z,c = self.get_input(batch, self.first_stage_key)
        N = z.shape[0]
        n_prompt = ""

        # STEP: Process each sample in the batch and put unconditional embedding if probabilty is true
        if self.p_cg:
            for idx in range(N):
                if self.cg_prng.choice(2, p=[self.p_cg, 1.-self.p_cg]):
                    print(f'Probability of {1. - self.p_cg} - using unconditional guidance')
                    c["c_crossattn"][idx] = torch.zeros_like(c["c_crossattn"][idx])
                    c["c_text"][idx] = self.get_learned_conditioning([n_prompt])
                    c["c_concat_mask"][idx] = torch.zeros_like(c["c_concat_mask"][idx])
            # STEP: Perform the forward pass and logs the metrics
            loss, loss_dict = self(z,c)

        else:
            loss, loss_dict = self.shared_step(batch)

        # on_epoch logs the metrics at the end of every epoch where the metrics are averaged out
        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)
        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)
        
        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss

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
            
            if 'c_src' in cond.keys():
                src = torch.cat(cond['c_src'], 1) # for bias, to be used in decoder
                biases = self.src_encoder(src)
                biases = [b*self.bias_scale for b in biases]
                for b in biases:
                    print(b.shape)
                eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, bias=biases, only_mid_control=self.only_mid_control)

            # STEP: If IP-Adapter is being used here, we concatenate them along the same dimension and chunk them for processing
            # concatenate by torch.cat((cond_text,cond_img), dim=1)
            else:
                eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)

        return eps

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):

        return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def log_images(self, batch, N=2, n_row=2, sample=False, ddim_steps=40, ddim_eta=0.0, 
                   plot_denoise_rows=False, plot_diffusion_rows=False, unconditional_guidance_scale=12.0,
                   log_ucg_ddimsteps_grid=False, ucg_values=None, ddim_steps_values=None,**kwargs):
        use_ddim = ddim_steps is not None

        log = dict()
        
        # NOTE: The image being fed into the VAE to get the latents is the target image
        # NOTE: The pose comes from the target image as well
        # NOTE: The style comes from the target image as well

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

            # reconstructed shape: [min(bs,N),3,512,512], x_sample_cfg: [min(bs,N),3,512,512]
            # NOTE: In this case, we only need the reconstructed and generated samples
            log['concat'] = torch.cat((reconstructed, x_samples_cfg), dim=-2)
            
        if log_ucg_ddimsteps_grid:
            assert ucg_values and ddim_steps_values, "ucg_values and ddim_steps values must be provided to plot grid."

            # STEP: Generate images
            grid_images = []
            for ucg in ucg_values:
                for step in ddim_steps_values:
                    # STEP: Get predicted original images
                    # this returns a batch of images using the current ucg and ddim_step parameter
                    sample_cfg, _ = self.sample_log(cond=cond, 
                                                     batch_size=N, ddim=use_ddim,
                                                     ddim_steps=step, eta=ddim_eta,
                                                     unconditional_guidance_scale=ucg,
                                                     unconditional_conditioning=un_cond
                                                    )
                    x_sample_cfg = self.decode_first_stage(sample_cfg) # shape: [B,C,H,W]
                    grid_images.append(x_sample_cfg) # list of tensors with shape [B,C,H,W]

            grid_images = torch.stack(grid_images, dim=0) # shape [NC,B,C,H,W] where NC is the number of combinations of parameters to test for
            grid_images = grid_images.permute(1,0,2,3,4) # shape [B, NC, C, H, W]
            # hee, we get a grid with NC number of images with NC being the number of combination of parameters for each sample in the batch
            log['grids'] = grid_images

        return log

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):

        b, c, h, w = cond["c_concat"][0].shape
        shape = (self.channels, h // 8, w // 8)
        # TODO: Need to find a way to pass in the source image (as a tensor ready for processing by VAE) as part of the conditions too
        samples, intermediates = self.ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        return samples, intermediates

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        
        # STEP: Determine whether to predict a grid of images for the different parameters
        log_ucg_ddimsteps_grid = False
        ucg_values = [10., 12.5, 15., 17.5, 20.]
        ddim_steps_values = [40, 80, 120, 160, 200]
        # we then make the grid and save it into files in the test_step function
        # we also provide the labelling of the grid in the test_step function

        # STEP: create folder
        f_ext = 'png'
        sample_root = Path(self.logger.save_dir)/'samples'
        gt_root = Path(self.logger.save_dir)/'gt'
        #mask_root = Path(self.logger.save_dir)/'mask'
        src_root = Path(self.logger.save_dir)/'src'
        concat_root = Path(self.logger.save_dir)/'concat'

        for root_name in [sample_root, gt_root, src_root, concat_root]:
            os.makedirs(str(root_name), exist_ok=True)        
        
        # STEP: inference
        images = self.log_images(batch, N=len(batch), ddim_steps=100, 
                                 unconditional_guidance_scale=14.0, sample=True,
                                 log_ucg_ddimsteps_grid=log_ucg_ddimsteps_grid, ucg_values=ucg_values,
                                 ddim_steps_values=ddim_steps_values)
         
        images['samples'] = torch.clamp(images['samples'].detach().cpu() * 0.5 + 0.5, 0., 1.1)
        images['samples']/=(torch.max(torch.abs(images['samples'])))

        for k in ['src_img', 'jpg']:
            batch[k] = (rearrange(batch[k],'b h w c -> b c h w') + 1.0) / 2.0

        # STEP: save samples
        for sample, fname, src_image, gt in \
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

        # STEP: save grids if we choose to
        if log_ucg_ddimsteps_grid:

            grid_root = Path(self.logger.save_dir)/'grid'
            os.makedirs(str(grid_root), exist_ok=True)

            # normalise values into the range of [0,1]            
            images['grids'] = torch.clamp(images['grids'].detach().cpu() * 0.5 + 0.5, 0., 1.)
            images['grids'] /= (torch.max(torch.abs(images['grids'])))
            
            # get labels of parameter combinations
            # NOTE: not in use currently
            labels = []
            for ucg_scale in ucg_values:
                for steps in ddim_steps_values:
                    labels.append(f"ucg={ucg_scale}, ddim={steps}")

            for grid, fname in zip(images["grids"], batch['fname']):
                grid = torch.clamp(grid*255., min=0., max=255.) # to convert grid to range of [0,255]
                img_grid = make_grid(grid, nrow=len(ucg_values), padding=5, pad_value=1.0)
                img_grid_np = img_grid.permute(1,2,0).detach().cpu().numpy()
                img_grid_np = img_grid_np.astype(np.uint8)
                
                filename=f'{fname}.{f_ext}'
                path = os.path.join(grid_root, filename)
                Image.fromarray(img_grid_np).save(path)

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.control_model.parameters())
        params += list(self.control_cond_model.parameters())
        if self.use_bias:
            params += list(self.src_encoder.parameters())
        if self.use_lora:
            lora_params = [p for n, p in self.model.named_parameters() if 'lora' in n]
            params += lora_params
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

