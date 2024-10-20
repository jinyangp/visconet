import torch.nn.functional as F

def resize_img_tensor(img_tensor,
                      resized_height:int,
                      resized_width:int):
    
    while len(img_tensor.shape) < 4:
        img_tensor = img_tensor.unsqueeze(0)

    return F.interpolate(img_tensor.float(), size=((resized_height, resized_width)), mode='bilinear', align_corners=False)
