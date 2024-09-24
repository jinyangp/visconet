import os
import torch

def load_adapter_weights(sd_model_state_dict,
                      adapter_state_dict=None,
                      adapter_ckpt_path=None):
    
    if not adapter_state_dict:
        assert adapter_ckpt_path, "Either adapter's state dictionary or checkpoint path has to be provided."
        adapter_state_dict = torch.load(adapter_ckpt_path)

        # TODO: Incorporate parts for image_proj
        image_proj_sd, ip_adapter_sd = adapter_state_dict["image_proj"], adapter_state_dict["ip_adapter"]
    
    # STEP: Load in weights of IP-Adapter layers
    '''
    TODO: BUG: size mismatch for model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_k_ip.weight: copying a param with shape torch.Size([320, 768]) from checkpoint, the shape in current model is torch.Size([320, 1024]).
    '''
    # 1.
    sd_model_state_dict['model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_k_ip.weight'] = ip_adapter_sd['1.to_k_ip.weight']
    sd_model_state_dict['model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_v_ip.weight'] = ip_adapter_sd['1.to_v_ip.weight']
    # 3.
    sd_model_state_dict['model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_k_ip'] = ip_adapter_sd['3.to_k_ip.weight']
    sd_model_state_dict['model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_v_ip'] = ip_adapter_sd['3.to_v_ip.weight']
    # 5.
    sd_model_state_dict['model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_k_ip'] = ip_adapter_sd['5.to_k_ip.weight']
    sd_model_state_dict['model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_v_ip'] = ip_adapter_sd['5.to_v_ip.weight']
    # 7.
    sd_model_state_dict['model.diffusion_model.input_blocks.5.1.transformer_blocks.0.attn2.to_k_ip'] = ip_adapter_sd['7.to_k_ip.weight']
    sd_model_state_dict['model.diffusion_model.input_blocks.5.1.transformer_blocks.0.attn2.to_v_ip'] = ip_adapter_sd['7.to_v_ip.weight']
    # 9.
    sd_model_state_dict['model.diffusion_model.input_blocks.7.1.transformer_blocks.0.attn2.to_k_ip'] = ip_adapter_sd['9.to_k_ip.weight']
    sd_model_state_dict['model.diffusion_model.input_blocks.7.1.transformer_blocks.0.attn2.to_v_ip'] = ip_adapter_sd['9.to_v_ip.weight']
    # 11.
    sd_model_state_dict['model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn2.to_k_ip'] = ip_adapter_sd['11.to_k_ip.weight']
    sd_model_state_dict['model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn2.to_v_ip'] = ip_adapter_sd['11.to_v_ip.weight']
    # 13.
    sd_model_state_dict['model.diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_k_ip.weight'] = ip_adapter_sd['13.to_k_ip.weight']
    sd_model_state_dict['model.diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_v_ip.weight'] = ip_adapter_sd['13.to_v_ip.weight']
    # 15.
    sd_model_state_dict['model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn2.to_k_ip.weight'] = ip_adapter_sd['15.to_k_ip.weight']
    sd_model_state_dict['model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn2.to_v_ip.weight'] = ip_adapter_sd['15.to_v_ip.weight']
    # 17.
    sd_model_state_dict['model.diffusion_model.output_blocks.4.1.transformer_blocks.0.attn2.to_v_ip.weight'] = ip_adapter_sd['17.to_k_ip.weight']
    sd_model_state_dict['model.diffusion_model.output_blocks.4.1.transformer_blocks.0.attn2.to_v_ip.weight'] = ip_adapter_sd['17.to_v_ip.weight']
    # 19.
    sd_model_state_dict['model.diffusion_model.output_blocks.5.1.transformer_blocks.0.attn2.to_v_ip.weight'] = ip_adapter_sd['19.to_k_ip.weight']
    sd_model_state_dict['model.diffusion_model.output_blocks.5.1.transformer_blocks.0.attn2.to_v_ip.weight'] = ip_adapter_sd['19.to_v_ip.weight']
    # 21.
    sd_model_state_dict['model.diffusion_model.output_blocks.6.1.transformer_blocks.0.attn2.to_v_ip.weight'] = ip_adapter_sd['21.to_k_ip.weight']
    sd_model_state_dict['model.diffusion_model.output_blocks.6.1.transformer_blocks.0.attn2.to_v_ip.weight'] = ip_adapter_sd['21.to_v_ip.weight']
    # 23.
    sd_model_state_dict['model.diffusion_model.output_blocks.7.1.transformer_blocks.0.attn2.to_v_ip.weight'] = ip_adapter_sd['23.to_k_ip.weight']
    sd_model_state_dict['model.diffusion_model.output_blocks.7.1.transformer_blocks.0.attn2.to_v_ip.weight'] = ip_adapter_sd['23.to_v_ip.weight']
    # 25.
    sd_model_state_dict['model.diffusion_model.output_blocks.8.1.transformer_blocks.0.attn2.to_v_ip.weight'] = ip_adapter_sd['25.to_k_ip.weight']
    sd_model_state_dict['model.diffusion_model.output_blocks.8.1.transformer_blocks.0.attn2.to_v_ip.weight'] = ip_adapter_sd['25.to_v_ip.weight']
    # 27.
    sd_model_state_dict['model.diffusion_model.output_blocks.9.1.transformer_blocks.0.attn2.to_v_ip.weight'] = ip_adapter_sd['27.to_k_ip.weight']
    sd_model_state_dict['model.diffusion_model.output_blocks.9.1.transformer_blocks.0.attn2.to_v_ip.weight'] = ip_adapter_sd['27.to_v_ip.weight']
    # 29.
    sd_model_state_dict['model.diffusion_model.output_blocks.10.1.transformer_blocks.0.attn2.to_v_ip.weight'] = ip_adapter_sd['29.to_k_ip.weight']
    sd_model_state_dict['model.diffusion_model.output_blocks.10.1.transformer_blocks.0.attn2.to_v_ip.weight'] = ip_adapter_sd['29.to_v_ip.weight']
    # 31.
    sd_model_state_dict['model.diffusion_model.output_blocks.11.1.transformer_blocks.0.attn2.to_v_ip.weight'] = ip_adapter_sd['31.to_k_ip.weight']
    sd_model_state_dict['model.diffusion_model.output_blocks.11.1.transformer_blocks.0.attn2.to_v_ip.weight'] = ip_adapter_sd['31.to_v_ip.weight']

    return sd_model_state_dict

def get_adapter_mask(mask_image):
    return 1 - mask_image