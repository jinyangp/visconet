import os
from huggingface_hub import upload_file

if __name__ == "__main__":

    # Replace this with your actual repo ID
    repo_id = "jinyangp/visconet2"
    config_file_root_dir = os.path.join(os.getcwd(), "configs")
    model_ckpt_root_dir = os.path.join(os.getcwd(),  "models")

    # Upload config files from 'configs' folder
    for config_file in ["visconet_baseline_baselinetest.yaml", "visconet_v8_pair_nullconditioning.yaml", "visconet_v8_pair.yaml", "visconet_v23_pair.yaml", "visconet_v24_pair.yaml"]:
        upload_file(
            path_or_fileobj=os.path.join(config_file_root_dir, config_file),  # Local path to the config file
            path_in_repo=f"configs/{config_file}",  # Path where it will be stored in the repo
            repo_id=repo_id,
    )
        
    # Upload ckpt  files from 'models' folder
    for model_file in ["visconet_v1.pth", "ablation1-visconet_v8_nullconditioning-gs55k.ckpt", "ablation2-visconet_v8_pair-gs155k.ckpt", "ablation3-visconet_v23_pair-gs100k.ckpt", "ablation4-visconet_v24_pair-gs25k.ckpt"]:
        upload_file(
            path_or_fileobj=os.path.join(model_ckpt_root_dir, model_file),  # Local path to the config file
            path_in_repo=f"models/{model_file}",  # Path where it will be stored in the repo
            repo_id=repo_id,
    )