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
            
            # STEP: Get the segmentation mask of fashion attributes
            styles_path = source["styles"]
            # two circumstances - we did not have the seg image fp or the seg image fp does not exist
            if pd.isna(styles_path) or not os.path.exists(str(self.style_root/styles_path)):
                full_styles_pil = None 
            else:
                full_styles_path = str(self.style_root/styles_path)
                full_styles_pil = Image.open(full_styles_path)

            # STEP: target - get ground truth and pose
            target = self.map_df.loc[row['to']]           
            target_path = str(self.image_root/target.name)
            # keep one pil to be passed to openpose to get pose, while the tensor is passed into the model as the groundtruth
            target_pil = Image.open(target_path)

            # STEP: get openpose pose (use pretrained from HF)
            target_pose = get_openpose_annotations(target_pil)
            pose_image = self.skeleton_tform(target_pose)

            target_image = self.image_tform(target_pil)

            # STEP: Get text prompt
            prompt = "a person."

            return dict(jpg=target_image, # previously, use the target image as the latents, now, we use the soruce image as the latents
                        #jpg=source_image,
                        txt=prompt,
                        hint=pose_image,
                        src_img=source_image,
                        src_img_pil=source_image_pil, # convert to numpy array, to be converted back to PIL image later
                        seg_img_pil=full_styles_pil, # convert to numpy array, to be converted back to PIL image later
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

# class DeepFashionDataset(Loader):
#     def __init__(self,
#                  image_root,
#                  image_dir,
#                  pose_dir,
#                  style_dir,
#                  mask_dir,
#                  map_file,
#                  data_files:list,
#                  dropout=None,
#                  sample_ratio=None,
#                  style_postfix='',
#                  image_shape=[512, 512],
#                  style_emb_shape=[1, 768],
#                  style_names=[],
#                  **kwargs):
#         super().__init__(**kwargs)
#         self.dropout = dropout
#         self.root = Path(image_root) # root directory
#         self.image_root = self.root/image_dir # source images
#         self.pose_root = self.root/pose_dir # pose images
#         self.style_root = self.root/style_dir # style images
#         self.mask_root = self.root/mask_dir # mask images
#         self.map_df = pd.read_csv(map_file) 
#         self.map_df.set_index('image', inplace=True)
#         dfs = [pd.read_csv(f) for f in data_files]
#         self.df = pd.concat(dfs, ignore_index=True)
#         self.style_postfix = style_postfix
#         if sample_ratio:
#             self.df = self.df.head(int(sample_ratio*len(self.df)))
#         # transformation # TODO: Need to reshape source images to 256x256?
#         self.image_tform = T.Compose([
#             T.Resize(image_shape),
#             T.ToTensor(),
#             T.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
        
#         self.skeleton_tform = T.Compose([
#             T.Resize(image_shape),
#             T.ToTensor(),
#             T.Lambda(lambda x: rearrange(x, 'c h w -> h w c'))])        
        
#         self.clip_norm = T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), 
#                                     std=(0.26862954, 0.26130258, 0.27577711))
#         self.clip_transform = T.Compose([
#             T.ToTensor(),
#             self.clip_norm
#         ])

#         self.style_names = style_names
#         self.style_emb_shape = style_emb_shape


#     def __len__(self):
#         return len(self.df)
    
#     # so we want the styles from the source image and regenerate the target image with the pose and styles of the source image
#     def __getitem__(self, index):
#         try:
#             # self.df refers to csv with to and from mapping of source images
#             row = self.df.iloc[index]
#             fname = get_name(row['from'], row['to'])

#             # source - get fashion styles
#             # self.map_df refers to csv with the mappings
#             # find the index of that source image in self.map_df
#             source = self.map_df.loc[row['from']]
#             src_path = str(self.image_root/source.name)
#             source_image = self.image_tform(Image.open(src_path))

#             # get the image where we want to get the styles from
#             styles_path = source['styles']
#             if styles_path == np.nan:
#                 return self.skip_sample(index)
            
#             full_styles_path = self.style_root/source['styles']

#             style_embeddings = []
#             # load the styles from an embedding file? else we give it a zero vector of shape [1,768]
#             # TODO: We get this from our image segmentor
#             for style_name in self.style_names:
#                 f_path = full_styles_path/(f'{style_name}'+self.style_postfix+'.p')
#                 if f_path.exists():
#                     with open(f_path, 'rb') as file:
#                         style_emb = pickle.load(file)
#                 else:
#                     #style_emb = np.zeros((1,768), dtype=np.float16)
#                     style_emb = np.zeros(self.style_emb_shape, dtype=np.float32)
#                 style_embeddings.append(style_emb)
#             styles = torch.tensor(np.array(style_embeddings)).squeeze(-2)

#             # target - get ground truth and pose
#             target = self.map_df.loc[row['to']]           
#             target_path = str(self.image_root/target.name)
#             target_image = self.image_tform(Image.open(target_path))

#             ## pose
#             # TODO: We need to get the pose image too via HF
#             target_path = str(self.pose_root/target.name)
#             pose_image = self.skeleton_tform(Image.open(target_path))

#             prompt = 'a person.'
#             # TODO: We do not have files of the mask - Need to get it from our ImageSegmentor
#             mask = T.ToTensor()(Image.open(str(self.mask_root/target.name).replace('.jpg','_mask.png')))
            
#             # TODO: Modify what is returned here
#             # Previusly this,
#             return dict(jpg=target_image, 
#                         txt=prompt,
#                         hint=pose_image,
#                         styles=styles,
#                         human_mask=mask,
#                         src_img=source_image,
#                         fname=fname)
            
#             # Now, we just need:
#             '''
#             return dict(jpg=target_image, 
#                         txt=prompt, 
#                         src_img=source_image,
#                         hint=pose_image,
#                         seg_img=seg_image,
#                         fname=fname
#                         )

#                         # styles=styles, (computed from own module in real time)
#                         # human_mask=mask, (coputed from own module in real time)
#             '''
        
#         except Exception as e:            
#             print(f"Skipping index {index}", e)
#             #sys.exit()
#             return self.skip_sample(index)


# class DeepFashionDatasetNumpy():
#     def __init__(self,
#                  image_root,
#                  image_dir,
#                  pose_dir,
#                  style_dir,
#                  mask_dir,
#                  map_file,
#                  data_files:list,
#                  dropout=None,
#                  sample_ratio=None,
#                  style_postfix='',
#                  image_shape=[512, 512],
#                  style_emb_shape=[1, 768],
#                  style_names=[],
#                  **kwargs):

#         self.root = Path(image_root)
#         self.image_root = self.root/image_dir
#         self.pose_root = self.root/pose_dir
#         self.style_root = self.root/style_dir
#         self.mask_root = self.root/mask_dir
#         self.map_df = pd.read_csv(map_file)
#         self.map_df.set_index('image', inplace=True)
#         dfs = [pd.read_csv(f) for f in data_files]
#         self.df = pd.concat(dfs, ignore_index=True)
#         self.style_postfix = style_postfix    
#         caption_file = self.root/'captions.json'
#         self.texts = json.load(open(caption_file)) if caption_file else None
#         self.style_names = style_names
        
#     def convert_fname(self, long_name):
#         gender = 'MEN' if long_name[7:10]  == 'MEN' else 'WOMEN'

#         input_list = long_name.replace('fashion','').split('___')

#         if gender == 'MEN':
#             pattern = r'MEN(\w+)id(\d+)_(\d)(\w+)'
#         else:
#             pattern = r'WOMEN(\w+)id(\d+)_(\d)(\w+)'

#         output_list = [f'{gender}/{category}/id_{id_num[:8]}/{id_num[8:]}_{view_num}_{view_desc}' for (category, id_num, view_num, view_desc) in re.findall(pattern, ' '.join(input_list))]

#         return [f +'.jpg' for f in output_list]
    
    
#     def get(self, df_names):       
#         src_name, dst_name = self.convert_fname(df_names)
#         source = self.map_df.loc[src_name]
#         src_path = str(self.image_root/source.name)
#         source_image = np.array(Image.open(src_path))
        
#         styles_folder = self.style_root/source['styles']
#         viscon_images = {}

#         for style_name in self.style_names:
#             f_path = styles_folder/f'{style_name}.jpg'
#             viscon_images[style_name] = np.array(Image.open(f_path)) \
#                                         if os.path.exists(str(f_path)) else None
#         # target pose, mask, caption
#         target = self.map_df.loc[dst_name]           
#         target_path = str(self.image_root/target.name)
#         target_image = np.array(Image.open(target_path))

#         ## pose
#         target_path = str(self.pose_root/target.name)
#         pose_image = np.array(Image.open(target_path))
#         mask = np.array(Image.open(str(self.mask_root/target.name).replace('.jpg','_mask.png')))
        
#         return {'source_image': source_image,
#                'target_image': target_image, 
#                'pose_image': pose_image,
#                'mask_image': mask,
#                'viscon_image': viscon_images}
