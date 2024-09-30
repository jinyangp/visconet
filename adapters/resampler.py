# adapted from https://github.com/tencent-ailab/IP-Adapter/blob/main/ip_adapter/resampler.py
import math
import torch
import torch.nn as nn

def reshape_tensor(x, heads):
    
    '''
    Args:
        x: tensor, vectors to be reshaped of shape [batch, seq_len, inner_dim]
        heads: int, number of attention heads to reshape over

    Returns:
        reshaped tensor of shape [batch, num_heads, seq_len, dim_per_head]
    '''

    bs, seq_len, _ = x.shape
    # (bs, length, inner_dim) --> (bs, length, n_heads, dim_per_head)
    x = x.view(bs, seq_len, heads, -1)
    # (bs, length, n_heads, dim_per_head) --> (bs, n_heads, length, dim_per_head)
    x = x.transpose(1,2)
    # (bs, n_heads, length, dim_per_head) --> (bs, n_heads, length, dim_per_head)
    x = x.reshape(bs, heads, seq_len, -1)
    
    return x


class FeedForward(nn.Module):
    
    def __init__(self,
                 dim,
                 mult=4):
        
        dim = dim
        inner_dim = dim*mult
        
        self.norm1 = nn.LayerNorm(dim)
        self.lin1 = nn.Linear(dim, inner_dim, bias=False)
        self.gelu = nn.GELU()
        self.lin2 = nn.Linear(inner_dim, dim, bias=False)

        self.dim = dim
        self.inner_dim = inner_dim

    def forward(self, x):
        
        x = self.norm1(x)
        x = self.lin1(x)
        x = self.gelu(x)
        x = self.lin2(x)
        
        return x


class PerceiverAttention(nn.Module):
    
    def __init__(self,
                 *,
                 dim,
                 dim_head=64,
                 heads=8):
        '''
        NOTE: The * means that no positional arguments are accepted and dim, dim_head and heads must be passed as keyword arguments
        '''

        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        '''
        Args:
            x (torch.Tensor): image features from image prompt
                shape (b, n1, D)
            latent (torch.Tensor): latent features, initially noise but we want to learn this and return as output
                shape (b, n2, D)
        '''
    
        # STEP: Normalise both x and latents
        x = self.norm1(x)
        latents = self.norm2(latents)
        
        # STEP: Save initial shape of latents
        batch, seq_len, _ = latents.shape

        # STEP: Pass x and latents to get Q,K,V vectors
        q = self.to_q(latents)

        # concatenate along sequence length dimension -> (batch, n1+n2, dim)
        kv_input = torch.cat([x, latents], dim=-2)
        # (batch, n1+n2, dim) -> (batch, n1+n2, inner_dim*2) -> (batch,n1+n2,inner_dim)
        k,v = self.to_kv(kv_input).chunk(2, dim=-1)
        
        # STEP: (batch, seq_len, inner_dim) --> (batch, heads, seq_len, dim_head) where inner_dim = head * dim_head
        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)
        
        # STEP: Perform attention mechanism
        scale = 1/math.sqrt(math.sqrt(self.dim_head))
        # (batch,heads,seq_len,dim_head) @ (batch,heads,dim_head,seq_len) --> (batch,heads,seq_len,seq_len)
        weights = (q*scale) @ (k*scale).transpose(-2,-1)
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v

        # STEP: Reshape back to input shape   
        # (batch,heads,length,dim_head) --> (bs, length,heads,dim_head) --> (bs,length,inner_dim)
        out = out.permute(0, 2, 1, 3).reshape(batch, seq_len, -1)
        return self.to_out(out)


class Resampler(nn.Module):
    
    def __init__(self,
                dim=1024, # NOTE: Dimensions the adapter works on
                depth=8, # NOTE: how many layers of PerceiverAttention and FeedForward stacked
                dim_head=64, # NOTE: Number of dimensions each head handles
                heads=16, # NOTE: Number of attention heads
                num_queries=32, # NOTE: Number of tokens in sequence each attribute should take up, 8 fashion attributes and 4 tokens per attribute
                embedding_dim=768, # NOTE: Dimensions of embeddings by image encoder
                output_dim=1024, # NOTE: Dimension of output embeddings
                ff_mult=4, # NOTE: Scale factor to multiply inner dim of FF by in the Resampler
                max_input_seq_len:int=257, # NOTE: Maximum sequence length of input token embeddings
                max_output_seq_len:int=77,
                apply_pos_emb: bool=False
    ):
        
        super().__init__()
        
        self.pos_emb = nn.Embedding(max_input_seq_len, embedding_dim) if apply_pos_emb else None

        # creates latents sampled from N(0,1) of shape (1,num_queries,dim) to learn about the original 1024 embedding
        # NOTE: This is what we return at the end
        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / dim**0.5)

        self.proj_in = nn.Linear(embedding_dim, dim)

        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                    FeedForward(dim=dim, mult=ff_mult)
                ])
            )
        
        self.max_output_seq_len = max_output_seq_len
    
    def forward(self, x):
    
        '''
        Args:
            x: tensor of shape [bs, num_attrs, 257, 1024] or [bs, num_attrs, 257, 768] depending if CLIP or DINO is used
            as the image encoder
        Returns:
            tensor of shape [bs,8*num_queries, output_dim]
        '''

        # STEP: Add positional embeddings if any
        bs, seq_len, device = x.shape[0], x.shape[1], x.device

        if self.pos_emb is not None:
            pos_emb = self.pos_emb(torch.arange(seq_len), device=device)
            x = x + pos_emb

        # STEP: Repeat the latents
        # (1,num_queries,dim) -> (bs,num_queries,dim)
        latents = self.latents.repeat(bs,1,1)

        # STEP: Project in using self.proj_in
        x = self.proj_in(x)

        # STEP: pass through attn and ff layers
        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents

        # STEP: Project out using self.proj_out and normalise
        latents = self.proj_out(latents)
        return self.norm_out(latents)