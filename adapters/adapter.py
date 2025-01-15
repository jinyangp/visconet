import torch
import torch.nn as nn

from adapters.resampler import Resampler

class Adapter:

    '''
    # TODO: Do not need this for now unless we are planning to implement IP-Adapter and IP-Adapter plus where inheritance is required.
    For now, we will only include the ImageProjModel of the IP-Adapter into the pipeline to generate more detailed embeddings.
    The IP Cross Attention layers can be implemented later.
    '''

    def __init__(self,
                 ckpt_path: str,
                 max_seq_len:int = 77,
                 num_tokens=4):
        
        '''
        Args:
            ckpt_path: str, path to checkpoint file
            num_tokens: int, number of tokens to generate for each token in the original embedding sequence
            i.e., given an initial embedding shape of [1, N, 257, 1024] --> [1, N*num_tokens, 257, 1024]
        '''

        def init_proj(self):
            # TODO: Fill this up
            self.image_proj_model = Resampler()


        