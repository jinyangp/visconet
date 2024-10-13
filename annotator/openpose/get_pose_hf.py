import os

from controlnet_aux import OpenPoseDetector # TODO: pip install controlnet-aux
from PIL import Image

def get_openpose_annotations(openpose_model_path: str,
                             img: Image.Image,
                             output_dir: str = None,
                             img_filename:str = None
                             ):
    
    openpose = OpenPoseDetector.from_pretrained("lllyasviel/ControlNet")
    openpose_image = openpose(img)

    if img_filename and output_dir:
                            
        full_output_dir = os.path.join(os.getcwd(), output_dir)
        if not os.path.exists(full_output_dir):
            os.makedirs(full_output_dir)

        output_full_path = os.path.join(full_output_dir, f'{img_filename}_openpose.jpg')
        openpose_image.save(output_full_path)

    return openpose_image