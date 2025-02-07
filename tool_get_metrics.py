import os
import torch
import argparse
import pandas as pd

from PIL import Image
from tqdm import tqdm
from visconet.metrics import image_tform, calculate_ssim, calculate_lpips, calculate_fid

def convert_fname(x):
    a, b = os.path.split(x)
    i = b.rfind('_')
    x = a + '/' +b[:i] + b[i+1:]
    return 'fashion'+x.split('.jpg')[0].replace('id_','id').replace('/','')

def get_name(src, dst):
    src = convert_fname(src)
    dst = convert_fname(dst)
    return src + '___' + dst

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Description of your program')

    '''
    srun -p rtx3090_slab -n 1 --job-name=test --gres=gpu:1 --kill-on-bad-exit=1 python3 -u tool_get_metrics.py ./data/deepfashion/benchmark-test-pairs-metrics.csv logs/011224-expt1-baseline-final-metrics logs 011224-expt1-metrics visconet_v5_pair --gpus 0
    '''
    
    # Adding arguments
    parser.add_argument('data_csv_fp', type=str)
    parser.add_argument("samples_dir", type=str)
    parser.add_argument('output_dir', type=str)
    parser.add_argument('name', type=str)
    parser.add_argument('config', type=str, help='config file')
    parser.add_argument('--gpus', nargs='+', type=int, default=[1])
   
     # STEP: Load in the inputs
    # Parsing arguments
    args = parser.parse_args()
    run_name = args.name
    data_csv_fp = os.path.join(os.getcwd(), args.data_csv_fp)
    data_df = pd.read_csv(data_csv_fp)
    samples_dir = os.path.join(os.getcwd(), args.samples_dir)
    f_ext = "png"

    # Collect all ground truth and generated samples
    all_src = []
    all_samples = []

    # STEP: Iterate through the csv file to collect all images
    for idx, row in tqdm(data_df.iterrows(), total=len(data_df), desc="Loading images"):
        fname = get_name(row['from'], row['to'])
        src_path = os.path.join(samples_dir, "src", f'{fname}.{f_ext}') # NOTE: Need src image in the "src" directory
        sample_path = os.path.join(samples_dir, "samples", f'{fname}.{f_ext}')
        
        # Load and transform images
        src = image_tform(Image.open(src_path)).to(torch.float32)
        sample = image_tform(Image.open(sample_path)).to(torch.float32)
        
        # Collect images
        all_src.append(src)
        all_samples.append(sample)

    # Convert collected images into tensors
    all_src = torch.stack(all_src, dim=0)  # Shape: (N, C, H, W)
    all_samples = torch.stack(all_samples, dim=0)  # Shape: (N, C, H, W)

    print(f"All SRC shape: {all_src.shape}, All samples shape: {all_samples.shape}")

    # Calculate SSIM
    ssim = calculate_ssim(all_src, all_samples).item()
    # Calculate LPIPS
    lpips = calculate_lpips(all_src, all_samples, rescale=True).item()
    # Calculate FID (ensure inputs are in uint8 if required by the function)
    fid = calculate_fid(all_src.to(torch.uint8), all_samples.to(torch.uint8)).item()

    res = {"run_name": [args.name],
           "config_name": [args.config],
           "ssim": ssim,
           "lpips": lpips,
           "fid": fid
           }
    res_df = pd.DataFrame(data=res)
    save_dir = os.path.join(os.getcwd(), args.output_dir)
    os.makedirs(save_dir, exist_ok=True)
    save_fp = os.path.join(save_dir, f'{args.name}.csv')
    res_df.to_csv(save_fp, index=False)
        
        



