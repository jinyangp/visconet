import os
import argparse
from share import *
import torch
from datetime import timedelta
from omegaconf import OmegaConf
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
from visconet.deepfashion import DeepFashionDataset, custom_collate_fn
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from ldm.modules.attention import CrossAttention
from ldm.util import instantiate_from_config
from visconet.styles_logger import StylesLogger
from pytorch_lightning.loggers import TensorBoardLogger

DEFAULT_CKPT = './models/visconet_v1.pth'

class SetupCallback(Callback):  
    def __init__(self, logdir, ckptdir, cfgdir, config):
        super().__init__()
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config

    def on_keyboard_interrupt(self, trainer, pl_module):
        if trainer.global_rank == 0:
            print("Summoning checkpoint.")
            ckpt_path = os.path.join(self.ckptdir, "last.ckpt")
            trainer.save_checkpoint(ckpt_path)

    def on_pretrain_routine_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            print("Project config")
            print(OmegaConf.to_yaml(self.config))
            OmegaConf.save(self.config,
                           os.path.join(self.cfgdir, "project.yaml"))


def main(args):

    model_config = args.config
    resume_path = args.resume_path
    gpus = args.gpus
    proj_name = args.name
    max_epochs = args.max_epochs
    batch_size = args.batch_size
    grad_acc = args.grad_acc
    num_gpus = len(gpus)
    num_workers = num_gpus * batch_size

    logdir = os.path.join('./logs/', proj_name)

    # logger_freq = 500
    learning_rate = num_gpus * (batch_size / 4) * 5e-5
    sd_locked = True
    only_mid_control = False
    
    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model(model_config).cpu()
    model.load_state_dict(load_state_dict(resume_path, location='cpu'), strict=False)
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control

    # NOTE: In newest iteration, we never reset the cross attention layer weights
    reset_crossattn = False
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

    # data
    dataset = instantiate_from_config(config.dataset.train)
    val_dataset = instantiate_from_config(config.dataset.val)
    dataloader = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, collate_fn=custom_collate_fn, shuffle=True, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, num_workers=num_workers, batch_size=batch_size, collate_fn=custom_collate_fn, shuffle=True, pin_memory=True)
    # set to False if want to investigate repeated generation over the same image
        
    # callbacks
    # NOTE: Determine frequency of train and validation batch frequency
    num_training_batches = len(dataloader)
    train_batch_freq = num_training_batches // 4
    num_val_batches = len(val_dataloader)
    val_batch_freq = num_val_batches // 4

    logger = ImageLogger(train_batch_frequency=train_batch_freq, val_batch_frequency=val_batch_freq)
    styles_logger = StylesLogger(train_batch_frequency=train_batch_freq, val_batch_frequency=val_batch_freq)
    setup_cb = SetupCallback(logdir=logdir, ckptdir=logdir, cfgdir=logdir, config=config)
    save_cb = ModelCheckpoint(dirpath=logdir,
                            save_last=True,
                            every_n_train_steps=8000, 
                            monitor='val/loss_simple_ema')
    lr_monitor_cb = LearningRateMonitor(logging_interval='step')
    callbacks = [logger, styles_logger, save_cb, setup_cb, lr_monitor_cb]

    # strategy = "ddp" if num_gpus > 1 else "auto"
    if num_gpus > 1:
        trainer = pl.Trainer(accelerator="gpu", devices=gpus, strategy="ddp",
                        precision=32,
                        callbacks=callbacks, 
                        accumulate_grad_batches=grad_acc,
                        default_root_dir=logdir,
                        # val_check_interval=1.0,
                        # val_check_interval=8000,
                        check_val_every_n_epoch=3,
                        num_sanity_val_steps=1,
                        max_epochs=max_epochs,
                        )
    else:
        trainer = pl.Trainer(accelerator="gpu", devices=gpus,
                        precision=32,
                        callbacks=callbacks, 
                        accumulate_grad_batches=grad_acc,
                        default_root_dir=logdir,
                        # val_check_interval=1.0,
                        # val_check_interval=8000,
                        check_val_every_n_epoch=3,
                        num_sanity_val_steps=1,
                        max_epochs=max_epochs,
                        )

    # Train!
    trainer.fit(model, dataloader, val_dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your program')

    # Adding arguments
    parser.add_argument('--name', type=str)
    parser.add_argument('--config', type=str, help='config file')
    parser.add_argument('--resume_path', type=str, default=DEFAULT_CKPT)
    parser.add_argument('--gpus', nargs='+', type=int, default=[1])
    parser.add_argument('--max_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--grad_acc', type=int, default=4)
    # Parsing arguments
    args = parser.parse_args()

    # Calling the main function with parsed arguments
    main(args)