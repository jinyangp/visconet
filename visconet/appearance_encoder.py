import torch
import torch.nn as nn

from einops import rearrange
from diffusers.models.attention import BasicTransformerBlock
from timm.models.swin_transformer import SwinTransformer

class AppearanceEncoder(nn.Module):

    def __init__(self,
                 in_channels,
                 model_channels,
                 depth=1,
                 embed_dims=256, # the context dim the transformer blocks should work on
                 context_dim=1024, # the context dim in the original UNET
                 channel_mult=[1,2,4],
                 transformer_depth=1,
                 num_heads=-1,
                 num_head_channels=-1, # set to 64
                 num_res_blocks=3,
                 get_featuremaps_method='conv'
                 ):

        '''
        Given original input of shape [1,3,512,512], VAE encodes it to latent
        space giving [1,4,64,64]
        Given latent input of shape [1,4,64,64] and the ControlNet settings of
        num_res_blocks = 2, channel_mult = [1,2,4,4] and model_channel =320
        
        Then, convolute to 320 channels and generate feature maps of [1,320,64,64], [1,640,32,32] and [1,1280,16,16]
        For each feature map, run them through zero conv and transformer blocks

        depth 1, block 1-3, input shape of [1,320,64,64]
        depth 2, block 1-3, input shape of [1,320,32,32]
        depth 3, block 1-3, input shape of [1,640,16,16]
        depth 4, block 1-3, input shape of [1,1280,8,8]
        '''
        super().__init__()
        
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.depth = depth
        self.num_res_blocks = num_res_blocks
        self.context_dim = context_dim
        if num_heads == -1:
           assert num_head_channels != -1, "Either num_heads or num_heads_channels must be provided."
        if num_head_channels != -1:
           assert embed_dims % num_head_channels == 0, "embed_dims must be perfectly divisible by num_head_channels."
        
        self.num_heads_channels = num_head_channels
        self.transformer_depth = transformer_depth
        self.channel_mult = channel_mult
        self.get_featuremaps_method = get_featuremaps_method

        conv_modules = [nn.Conv2d(in_channels, model_channels, kernel_size=3, stride=1, padding=1),
                        nn.Conv2d(model_channels, model_channels, kernel_size=3, stride=2, padding=1)]
        zero_convin = []
        zero_convout = []
        transformer_blocks = []

        ch = model_channels
        # make modules to get feature maps
        # TODO: For conv method, add normalisation and non-linear activation after every Conv layer
        if self.get_featuremaps_method == "conv":
           for mult_factor in channel_mult[1:]:
               conv_modules.append(nn.Conv2d(ch, model_channels*mult_factor, kernel_size=3, stride=2, padding=1))
               ch = model_channels*mult_factor
        elif self.get_featuremaps_method == "attn":
            conv_modules.append(SwinTransformer(img_size=32,
                                                patch_size=2,
                                                in_chans=320,
                                                embed_dim=640, # embed dim for each image patch
                                                depths=(2,1), # (num layers, num attn block per layer)
                                                num_heads=(8,16),
                                                window_size=8, # window_size for self attn
                                                mlp_ratio=4.0, # ratio of hidden dim to embed_dim
                                                qkv_bias=True, # add bias in attn block
                                                num_classes=0, # don't want classification
                                                downsample="merging"
                                                ))

        # make modules for zero conv in and conv out and transformer blocks
        ch = model_channels
        for mult_factor in channel_mult:
           zero_convin.extend([nn.Conv2d(ch, embed_dims*mult_factor, kernel_size=1, stride=1, padding=0) for _ in range(self.num_res_blocks)])
           zero_convout.extend([nn.Conv2d(embed_dims*mult_factor, self.context_dim, kernel_size=1, stride=1, padding=0) for _ in range(self.num_res_blocks)])          
           transformer_layers = nn.Sequential(*[BasicTransformerBlock(dim=embed_dims*mult_factor,
                                                            num_attention_heads=(embed_dims*mult_factor//self.num_heads_channels),
                                                            attention_head_dim=self.num_heads_channels,
                                                            double_self_attention=True)
                                                            for _ in range(self.transformer_depth)])
           transformer_blocks.extend([transformer_layers for _ in range(self.num_res_blocks)])
           ch = model_channels*mult_factor
        
        self.convs = nn.Sequential(*conv_modules)
        self.zero_convin = nn.ModuleList(zero_convin)
        self.zero_convout = nn.ModuleList(zero_convout)
        self.transformer_blocks = nn.ModuleList(transformer_blocks)

        for n in self.zero_convin.parameters():
           nn.init.zeros_(n)
        for n in self.zero_convout.parameters():
           nn.init.zeros_(n)

    def forward(self, x):

        # Get a list of all the outputs from the conv layers first
        feature_maps = []
        for layer in self.convs:
            if self.get_featuremaps_method == "attn" and not isinstance(layer, nn.modules.conv.Conv2d):
               _, xs = layer.forward_intermediates(x) # xs is a list of tensors
               for x in xs:
                  feature_maps.extend([x for _ in range(self.num_res_blocks)])                  
            else:
               x = layer(x)
               feature_maps.extend([x for _ in range(self.num_res_blocks)])

        outs = []
        for i in range(len(self.channel_mult)*self.num_res_blocks):
            
            conv_in = self.zero_convin[i]
            transformer_layers = self.transformer_blocks[i]
            conv_out = self.zero_convout[i]
            features = feature_maps[i]

            # shape: [B,C,H,W] -> [B,C,H,W]
            features = conv_in(features)
            for layer in transformer_layers:
                # features input: [B,C,H,W]
                # input shape required by transformer block: [B, H*W, C]
                # output shape required by transformer block: [B, H*W, C]
                b,c,h,w = features.shape
                # Rearrange features from [b, c, h, w] to [b, h*w, c]
                features = features.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)
                # Pass through the layer
                features = layer(features)
                # Rearrange features back from [b, h*w, c] to [b, c, h, w]
                features = features.view(b, h, w, c).permute(0, 3, 1, 2)
            features = conv_out(features)
            outs.append(features)

        return outs