import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from pathlib import Path
from PIL import Image
import pandas as pd
from abc import abstractmethod
import pickle
import numpy as np
from einops import rearrange
import json
import re

from annotator.openpose.get_pose_hf import get_openpose_annotations

class Loader(Dataset):
    def __init__(self, shuffle=False):
        super().__init__()
        self.shuffle = shuffle
    
    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)

    @abstractmethod
    def __getitem__(self, ind):
        pass

def convert_fname(x):
    a, b = os.path.split(x)
    i = b.rfind('_')
    x = a + '/' +b[:i] + b[i+1:]
    return 'fashion'+x.split('.jpg')[0].replace('id_','id').replace('/','')

def get_name(src, dst):
    src = convert_fname(src)
    dst = convert_fname(dst)
    return src + '___' + dst

def list_subdirectories(path):
    subdirectories = []
    for dirpath, dirnames, filenames in os.walk(path):
        if not dirnames:
            subdirectories.append(dirpath)
    return subdirectories

class DeepFashionDataset(Loader):

    def __init__(self,
                 image_root,
                 image_dir,
                 style_dir,
                 map_file,
                 data_files:list,
                 sample_ratio=None, # ratio of entire dataset to sample from
                 image_shape=[512, 512], # image shape to transform images to to match visconet expected inputs
                 **kwargs
                 ):
        
        super().__init__(**kwargs)
        self.root = Path(image_root)
        self.image_root = self.root/image_dir
        self.style_root = self.root/style_dir

        self.map_df = pd.read_csv(map_file)
        
        # STEP: Add column to file path of segmentation map (do this before we make image the index of the df and we can't reference it as a column anymore)
        def append_styles_fp_col(source_img_partial_fp):
            partial_fn = source_img_partial_fp.split(".")[0]
            partial_fn = partial_fn.replace("/", "-")
            partial_fn += "_segm.png" 
            styles_full_fp = os.path.join(self.style_root, partial_fn)

            if os.path.exists(styles_full_fp):
                return partial_fn
            else:
                return np.nan
        
        if "styles" not in self.map_df.columns:    
            self.map_df["styles"] = self.map_df["image"].apply(lambda fp: append_styles_fp_col(fp))
        
        self.map_df.set_index('image', inplace=True)

        dfs = [pd.read_csv(f) for f in data_files]
        self.df = pd.concat(dfs, ignore_index=True)
        if sample_ratio:
            self.df = self.df.head(int(sample_ratio*len(self.df)))

        self.image_tform = T.Compose([
            T.Resize(image_shape),
            T.ToTensor(),
            T.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
        
        self.skeleton_tform = T.Compose([
            T.Resize(image_shape),
            T.ToTensor(),
            T.Lambda(lambda x: rearrange(x, 'c h w -> h w c'))])        

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        
        try:
            row = self.df.iloc[index]
            fname = get_name(row['from'], row['to'])

            # STEP: get source image            
            source = self.map_df.loc[row['from']]
            src_path = str(self.image_root/source.name)

            # keep one pil and another in tensor, pil to be passed to localstyleprojector while tensor to be passed into model
            source_image_pil = Image.open(src_path)
            source_image = self.image_tform(Image.open(src_path))
            
            # STEP: target - get ground truth and pose
            target = self.map_df.loc[row['to']]           
            target_path = str(self.image_root/target.name)
            # keep one pil to be passed to openpose to get pose, while the tensor is passed into the model as the groundtruth
            target_pil = Image.open(target_path)

            # STEP: target - get the segmentation mask of fashion attributes
            styles_path = target["styles"]
            # two circumstances - we did not have the seg image fp or the seg image fp does not exist
            if pd.isna(styles_path) or not os.path.exists(str(self.style_root/styles_path)):
                full_styles_pil = None 
            else:
                full_styles_path = str(self.style_root/styles_path)
                full_styles_pil = Image.open(full_styles_path)

            # STEP: get openpose pose (use pretrained from HF)
            target_pose = get_openpose_annotations(target_pil)
            pose_image = self.skeleton_tform(target_pose)

            target_image = self.image_tform(target_pil)

            # STEP: Get text prompt
            prompt = "a person."

            # NOTE: We use the target image as jpg key (noisy latents passed to UNET) and also derive the styles, target pose and mask from target image
            # Goal is to reconstruct the target image using the segmented target styles, target pose and mask from the target image
            return dict(jpg=target_image, # tensor
                        # jpg=source_image,
                        txt=prompt, # text
                        hint=pose_image, # tensor
                        src_img=source_image, # tensor
                        src_img_pil=source_image_pil, # pil image
                        seg_img_pil=full_styles_pil, # pil image
                        target_img_pil=target_pil, # pil image
                        fname=fname)
        
        except Exception as e:            
            print(f"Skipping index {index}", e)
            #sys.exit()
            return self.skip_sample(index)
    
def custom_collate_fn(batch):
    
    collated_batch = {}
    
    for key in batch[0].keys():
        if key == "seg_img_pil":
            # Handle seg_img_pil: collect PIL images or None
            collated_batch[key] = [sample[key] for sample in batch]
        else:
            # For other keys, stack tensors
            collated_batch[key] = [sample[key] for sample in batch]
            if isinstance(collated_batch[key][0], torch.Tensor):
                collated_batch[key] = torch.stack(collated_batch[key])  # Stack tensors
    
    return collated_batch