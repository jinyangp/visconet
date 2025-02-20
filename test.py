import os
import argparse
from share import *
import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
#from tutorial_dataset import MyDataset
from visconet.deepfashion import DeepFashionDataset, custom_collate_fn
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from ldm.modules.attention import CrossAttention
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from visconet.styles_logger import StylesLogger

# NOTE: I changed the base path here
DEFAULT_CKPT = './models/visconet_v1.pth'

# Configs
#model_config = './models/cldm_v15.yaml'
#resume_path = './models/control_sd15_ini.ckpt'
#model_config = './configs/visconet_v21.yaml'
#model_config = './configs/pose_transfer.yaml'
#resume_path = './models/control_sd21_ini.ckpt'
#resume_path ='/home/soon/github/visconet/lightning_logs/sd21_styles_mask_1103/epoch=1-step=1722.ckpt'
#logdir = './lightning_logs/sd21_pose_mask_1103/'

def main(args):

    model_config = args.config
    resume_path = args.resume_path
    gpus = args.gpus
    batch_size = args.batch_size
    num_workers = len(gpus) * batch_size
    
    proj_name = args.name
    max_epochs = args.max_epochs

    logdir = os.path.join('./logs/', proj_name)

    if resume_path == '':
        resume_path = DEFAULT_CKPT
        reset_crossattn = True
    else:
        reset_crossattn = False
    
    logger_freq = 10000
    learning_rate = 5e-5
    sd_locked = True
    only_mid_control = False
    
    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model(model_config).cpu()
    model.load_state_dict(load_state_dict(resume_path, location='cpu'), strict=False)
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control

    # NOTE: Determine if we want to blur the mask
    if args.use_exact_mask:
        model.control_cond_model.human_segmentor.blur_mask = False

    # initialize cross attention weights
    if reset_crossattn:
        for name, module in model.control_model.named_modules():
            if isinstance(module, CrossAttention):
                print(f"Found CrossAttention Layer: {name}")
                # Reset parameters of the CrossAttention layer
                if hasattr(module, 'reset_parameters'):
                    with torch.no_grad():
                        for param in module.parameters():
                            module.reset_parameters()  # Reset the parameters of the CrossAttention layer

    config = OmegaConf.load(model_config)

    test_dataset = instantiate_from_config(config.dataset.test)
    test_dataloader = DataLoader(test_dataset, num_workers=num_workers, batch_size=batch_size, collate_fn=custom_collate_fn, shuffle=False, pin_memory=True)
    
    styles_logger = StylesLogger()
    callbacks = [styles_logger]

    trainer = pl.Trainer(
                        strategy="ddp",
                        accelerator="gpu", devices=gpus, 
                        precision=32,
                        callbacks=callbacks, 
                        accumulate_grad_batches=4,
                        default_root_dir=logdir,
                        check_val_every_n_epoch=1,
                        num_sanity_val_steps=1,
                        max_epochs=max_epochs)

    # Train!
    trainer.test(model, test_dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your program')

    # Adding arguments
    parser.add_argument('--name', type=str)
    parser.add_argument('--config', type=str, help='config file')
    parser.add_argument('--resume_path', type=str, default='')
    parser.add_argument('--gpus', nargs='+', type=int, default=[1])
    parser.add_argument('--max_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--use_exact_mask', action="store_true", help="Whether to use the exact mask of human segmentor.")
    
    # Parsing arguments
    args = parser.parse_args()

    # Calling the main function with parsed arguments
    main(args)