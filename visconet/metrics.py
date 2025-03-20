# We want to calculate the metrics for SSIM, FID, and LPIPS
import torch

from torchvision import transforms as T
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

image_tform = T.Compose([
    T.Resize((256,256)),
    T.PILToTensor() # Converts PIL to tensor in range of [0,1]
])

def calculate_ssim(preds,
                   targets,
                   transform=False,
                   aggregate='elementwise_mean'):
    '''
    NOTE: Expects values to be in the range of [0,1]
    Args:
        preds: tensor of shape [B,C,H,W] if transform is False else a list of PIL images
        targets: tensor of shape [B,C,H,W] if transform is False else a list of PIL images
        transform: flag to decide if neeed to convert to tensor and resize
        reduce: flag to decide if need to reduce the range from [0,1] to [-1,1]
        aggregate: str, aggregation method
    Returns:
        a list of scores if aggregate = None else the elementwise_mean of scores
        if reduction method is elementwise_mean
    '''
    if transform:
        preds = torch.stack([image_tform(img) for img in preds])
        targets = torch.stack([image_tform(img) for img in targets])

    # NOTE: 1. is the optimal value
    ssim = StructuralSimilarityIndexMeasure(reduction=aggregate)
    return ssim(preds,targets)

def calculate_lpips(preds,
                    targets,
                    transform=False,
                    rescale=False,
                    reduction='mean'
                    ):

    '''
    # NOTE: 
    NOTE: Expects values to be in the range of [-1,1]
    Args:
        preds: tensor of shape [B,C,H,W] if transform is False else a list of PIL images
        targets: tensor of shape [B,C,H,W] if transform is False else a list of PIL images
        transform: flag to decide if neeed to convert to tensor and resize
        rescale: flag to decide if need to reduce the range from [0,1] to [-1,1]
        reduction: mean or sum, to aggregate output
    
    Returns:
        aggregated LPIPS score
    '''

    if transform:
        preds = torch.stack([image_tform(img) for img in preds])
        targets = torch.stack([image_tform(img) for img in targets])

    if rescale:
        preds = preds * 2 - 1
        targets = targets * 2 - 1
    
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg', reduction=reduction)
    return lpips(preds, targets)

def calculate_fid(preds,
                  targets,
                  transform=False,
                  inception_features=64
                ):
    
    '''
    Args:
        preds: tensor of shape [B,C,H,W] if transform is False else a list of PIL images
        targets: tensor of shape [B,C,H,W] if transform is False else a list of PIL images
        transform: flag to decide if neeed to convert to tensor and resize
        inception_features: int, which inceptionv3 feature layer to choose

    Returns:
        float scalar tensor with mean FID value over samples
    '''

    if transform:
        preds = torch.stack([image_tform(img) for img in preds])
        targets = torch.stack([image_tform(img) for img in targets])
    
    fid = FrechetInceptionDistance(feature=inception_features)
    fid.update(preds,real=False)
    fid.update(targets, real=True)
    return fid.compute()