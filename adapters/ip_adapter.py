import os
import torch


def load_state_dict(ckpt_path: str):

    ckpt_fullpath = os.path.join(os.getcwd(), ckpt_path)
    assert os.path.exists(ckpt_fullpath), "Provided ckpt_path is invalid."
    ip_adapter_state_dict = torch.load(ckpt_path)

    return ip_adapter_state_dict

def load_adapter_weights(sd_model_state_dict,
                      adapter_state_dict=None,
                      adapter_ckpt_path=None):
    
    if not adapter_state_dict:
        assert adapter_ckpt_path, "Either adapter's state dictionary or checkpoint path has to be provided."
        adapter_state_dict = load_state_dict(adapter_ckpt_path)
    
    # 1.
    sd_model_state_dict['model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_k_ip.weight'] = adapter_state_dict['1.to_k_ip.weight']
    sd_model_state_dict['model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_v_ip.weight'] = adapter_state_dict['1.to_v_ip.weight']
    # 3.
    sd_model_state_dict['model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_k_ip'] = adapter_state_dict['3.to_k_ip.weight']
    sd_model_state_dict['model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_v_ip'] = adapter_state_dict['3.to_v_ip.weight']
    # 5.
    sd_model_state_dict['model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_k_ip'] = adapter_state_dict['5.to_k_ip.weight']
    sd_model_state_dict['model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_v_ip'] = adapter_state_dict['5.to_v_ip.weight']
    # 7.
    sd_model_state_dict['model.diffusion_model.input_blocks.5.1.transformer_blocks.0.attn2.to_k_ip'] = adapter_state_dict['7.to_k_ip.weight']
    sd_model_state_dict['model.diffusion_model.input_blocks.5.1.transformer_blocks.0.attn2.to_v_ip'] = adapter_state_dict['7.to_v_ip.weight']
    # 9.
    sd_model_state_dict['model.diffusion_model.input_blocks.7.1.transformer_blocks.0.attn2.to_k_ip'] = adapter_state_dict['9.to_k_ip.weight']
    sd_model_state_dict['model.diffusion_model.input_blocks.7.1.transformer_blocks.0.attn2.to_v_ip'] = adapter_state_dict['9.to_v_ip.weight']
    # 11.
    sd_model_state_dict['model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn2.to_k_ip'] = adapter_state_dict['11.to_k_ip.weight']
    sd_model_state_dict['model.diffusion_model.input_blocks.8.1.transformer_blocks.0.attn2.to_v_ip'] = adapter_state_dict['11.to_v_ip.weight']
    # 13.
    sd_model_state_dict['model.diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_k_ip.weight'] = adapter_state_dict['13.to_k_ip.weight']
    sd_model_state_dict['model.diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_v_ip.weight'] = adapter_state_dict['13.to_v_ip.weight']
    # 15.
    sd_model_state_dict['model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn2.to_k_ip.weight'] = adapter_state_dict['15.to_k_ip.weight']
    sd_model_state_dict['model.diffusion_model.output_blocks.3.1.transformer_blocks.0.attn2.to_v_ip.weight'] = adapter_state_dict['15.to_v_ip.weight']
    # 17.
    sd_model_state_dict['model.diffusion_model.output_blocks.4.1.transformer_blocks.0.attn2.to_v_ip.weight'] = adapter_state_dict['17.to_k_ip.weight']
    sd_model_state_dict['model.diffusion_model.output_blocks.4.1.transformer_blocks.0.attn2.to_v_ip.weight'] = adapter_state_dict['17.to_v_ip.weight']
    # 19.
    sd_model_state_dict['model.diffusion_model.output_blocks.5.1.transformer_blocks.0.attn2.to_v_ip.weight'] = adapter_state_dict['19.to_k_ip.weight']
    sd_model_state_dict['model.diffusion_model.output_blocks.5.1.transformer_blocks.0.attn2.to_v_ip.weight'] = adapter_state_dict['19.to_v_ip.weight']
    # 21.
    sd_model_state_dict['model.diffusion_model.output_blocks.6.1.transformer_blocks.0.attn2.to_v_ip.weight'] = adapter_state_dict['21.to_k_ip.weight']
    sd_model_state_dict['model.diffusion_model.output_blocks.6.1.transformer_blocks.0.attn2.to_v_ip.weight'] = adapter_state_dict['21.to_v_ip.weight']
    # 23.
    sd_model_state_dict['model.diffusion_model.output_blocks.7.1.transformer_blocks.0.attn2.to_v_ip.weight'] = adapter_state_dict['23.to_k_ip.weight']
    sd_model_state_dict['model.diffusion_model.output_blocks.7.1.transformer_blocks.0.attn2.to_v_ip.weight'] = adapter_state_dict['23.to_v_ip.weight']
    # 25.
    sd_model_state_dict['model.diffusion_model.output_blocks.8.1.transformer_blocks.0.attn2.to_v_ip.weight'] = adapter_state_dict['25.to_k_ip.weight']
    sd_model_state_dict['model.diffusion_model.output_blocks.8.1.transformer_blocks.0.attn2.to_v_ip.weight'] = adapter_state_dict['25.to_v_ip.weight']
    # 27.
    sd_model_state_dict['model.diffusion_model.output_blocks.9.1.transformer_blocks.0.attn2.to_v_ip.weight'] = adapter_state_dict['27.to_k_ip.weight']
    sd_model_state_dict['model.diffusion_model.output_blocks.9.1.transformer_blocks.0.attn2.to_v_ip.weight'] = adapter_state_dict['27.to_v_ip.weight']
    # 29.
    sd_model_state_dict['model.diffusion_model.output_blocks.10.1.transformer_blocks.0.attn2.to_v_ip.weight'] = adapter_state_dict['29.to_k_ip.weight']
    sd_model_state_dict['model.diffusion_model.output_blocks.10.1.transformer_blocks.0.attn2.to_v_ip.weight'] = adapter_state_dict['29.to_v_ip.weight']
    # 31.
    sd_model_state_dict['model.diffusion_model.output_blocks.11.1.transformer_blocks.0.attn2.to_v_ip.weight'] = adapter_state_dict['31.to_k_ip.weight']
    sd_model_state_dict['model.diffusion_model.output_blocks.11.1.transformer_blocks.0.attn2.to_v_ip.weight'] = adapter_state_dict['31.to_v_ip.weight']

    return sd_model_state_dict

def get_adapter_mask(mask_image):
    return 1 - mask_image