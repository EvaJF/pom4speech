"""This module mixes information from different tokens via Polynomial Mixing.
It can be viewed as a linear-time drop-in replacement for (self-)attention.

source: https://arxiv.org/abs/2411.12663 

Authors
 * ...
"""
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class PolynomialMixer(nn.Module):
    """This class implements multi-head Polynomial Mixing.
    It is an implementation of the token-mixing component in PoM, a linear
    time drop-in replacement for self-attention. 

    Reference: https://arxiv.org/abs/2411.12663 

    Arguments
    ---------   
    input_output_dim : int 
        The dimensionality of the input features / number of features in keys, queries, and values
    degree :int 
        The degree of the polynomial to be captured.
    expand : int 
        The expansion factor for the order.
    bias : bool (optional) 
        Whether to include a bias term in the linear projections. Default is True.
    num_heads : int (optional)
        number of parallel PoM operations
    max_length : int (optional)
        Maximum number of input tokens. Needed for generating sufficiently large position embeddings.

    Example
    -------
    >>> mixer = PolynomialMixer(input_output_dim=512, degree=3, expand=2)
    >>> query = torch.randn(8, 100, 512) # (Batch, Length, Embedding)
    >>> key = torch.randn(8, 100, 512) 
    >>> value = torch.randn(8, 100, 512) 
    >>> output, _ = mixer(query, key, value)
    >>> output.shape
    torch.Size([8, 100, 512])
    """

    def __init__(
        self,
        input_output_dim: int,
        degree: int,
        expand: int,
        bias: bool=True,
        num_heads: int = 1,
        max_length: int = 3000,
    ) -> None:
        super().__init__()
        self.input_output_dim = input_output_dim
        self.mh_pom = PoM(
            input_output_dim,
            degree,
            expand,
            bias = bias,
            num_heads=num_heads,
        )

        self.layer_norm = nn.LayerNorm(input_output_dim)

    def forward(
        self,
        query,
        key,
        value,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_attn_weights: Optional[bool] = True,
        pos_embs: Optional[torch.Tensor] = None,
    ):
        """
        The signature of this method is deliberately chosen to be the same as for
        sb.nnet.attention.MultiHeadAttention for compatibility within SpeechBrain.

        NOTE: argument key has no effect. 
        Query is used for the only necessary input. 
        If the query matrix is provided as context/value matrix, self-attention is performed. 

        Arguments
        ----------
        query : torch.Tensor
            (B, L, E) where L is the target sequence length,
            B is the batch size, E is the embedding dimension.
        key : torch.Tensor
            (B, S, E) where S is the source sequence length,
            B is the batch size, E is the embedding dimension.
            Currently unused. All
        value : torch.Tensor
            (B, S, E) where S is the source sequence length,
            B is the batch size, E is the embedding dimension.
            Currently unused.
        attn_mask : torch.Tensor, optional
            mask for masking the inputs to the token mixer
        key_padding_mask : torch.Tensor, optional
            (B, S) where B is the batch size, S is the source sequence
            length. If a ByteTensor is provided, the non-zero positions will
            be ignored while the position with the zero positions will be
            unchanged. If a BoolTensor is provided, the positions with the
            value of True will be ignored while the position with the value
            of False will be unchanged.
        return_attn_weights: torch.Tensor, optional
            NOTE: Currently has NO effect.
        pos_embs: torch.Tensor, optional
            NOTE: Currently has NO effect.

        Outputs
        -------
        attn_output : torch.Tensor
            (B, L, E) where L is the target sequence length, B is the
            batch size, E is the embedding dimension.
        attn_output_weights : torch.Tensor
            (B, L, S) where B is the batch size, L is the target
            sequence length, S is the source sequence length.
            NOTE: always returns all zeros.
        """    

        # NOTE: We are ignoring keys AND values for now (so, only self-attention is supported atm)
        out = query

        bsize = out.size(0)
        seq_len = out.size(1)

        if key_padding_mask is not None:
            float_mask = (
                torch.logical_not(key_padding_mask).unsqueeze(-1).float()
            )
            out = out * float_mask

        # add position embedding before PoM
        pom_input = out 
        
        # PoM
        pom_output = self.mh_pom(pom_input, 
                                 pom_input, 
                                 attn_mask) 
        #[bsize, seq_len, input_output_dim]
        
        # apply layer norm on outputs of the TM-MLP
        pom_output = self.layer_norm(pom_output)

        dummy_att_weights = torch.zeros(
            (bsize, seq_len, seq_len), device=out.device
        )
        return pom_output, dummy_att_weights
    

class PoM(nn.Module):
    """
    PoM (Polynomial Mixer) Module

    This class implements the PoM (Polynomial Mixer) module, which is a custom neural network layer
    designed for capturing higher-order interactions between input features. It consists of three
    linear projections and a custom PoM operation.

    Attributes:
        dim (int): The dimensionality of the input features.
        order (int): The order of the moments to be captured.
        order_expand (int): The expansion factor for the order.
        num_heads (int): The number of heads to compute in parallel. 
        po_proj (nn.Linear): Linear projection for the polynomials.
        se_proj (nn.Linear): Linear projection for the selection.
        ag_proj (nn.Linear): Linear projection for aggregating the results.
        pom (callable): The custom polynomial mixer operation function.

    Args:
        dim (int): The dimensionality of the input features.
        degree (int): The degree of the polynomial to be captured.
        expand (int): The expansion factor for the order.
        num_heads (int): The number of heads to compute in parallel. 
        bias (bool, optional): Whether to include a bias term in the linear projections. Default is True.
    """
    def __init__(self, dim, degree, expand, num_heads=1, bias=True):
        super().__init__()
        self.dim = dim
        self.order = degree
        self.order_expand = expand
        self.num_heads = num_heads

        self.po_proj_hf = nn.Linear(dim//2, degree * expand * dim//2, bias=bias)
        self.po_proj_bf = nn.Linear(dim-dim//2, degree * expand * (dim - dim//2), bias=bias)
        self.se_proj_hf = nn.Linear(dim//2, degree * expand * dim//2, bias=bias)
        self.se_proj_bf = nn.Linear(dim-dim//2, degree * expand * (dim - dim//2), bias=bias)
        self.ag_proj = nn.Linear(degree * expand * dim, dim, bias=bias)
        self.pom = pom

    def forward(self, xq, xc=None, mask=None):
        """
        Forward pass of the PoM module.

        Args:
            xq (torch.Tensor): Q; The query input tensor of size batch x n_tokens x dimension.
            xc (torch.Tensor, optional): V; The context input tensor. If None, self-attention is performed.
            mask (torch.Tensor, optional): The mask tensor for attention.

        Returns:
            torch.Tensor: The output tensor after applying the PoM operation.
        """
        if xc is None:
            xc = xq # self attention

        xq_hf = xq[:, :, :self.dim//2] # slice features 1 : dim//2
        xq_bf = xq[:, :, self.dim//2:] # slice features dim//2+1 : dim
        xc_hf = xc[:, :, :self.dim//2] # slice features 1 : dim//2
        xc_bf = xc[:, :, self.dim//2:] # slice features dim//2+1 : dim

        s_hf = self.se_proj_hf(xq_hf)
        s_bf = self.se_proj_bf(xq_bf)
        h_hf = self.po_proj_hf(xc_hf) 
        h_bf = self.po_proj_bf(xc_bf)
        sh_hf = self.pom(s_hf, h_hf, self.order, mask)
        sh_bf = self.pom(s_bf, h_bf, self.order, mask)
        sh = torch.cat([sh_hf, sh_bf], dim=-1)
        
        # aggregation : still interaction between hf and for the final projection
        return self.ag_proj(sh)
    
def pom(xq: torch.Tensor, xc: torch.Tensor, k: int, mask=None):
    """
    pom function

    This function implements the polynomial mixer operation.
            Args:
            xq (torch.Tensor): The query input tensor.
            xc (torch.Tensor): The context input tensor.
            k (int): The order of the polynomial.
            mask (torch.Tensor, optional): The mask tensor for attention.
    """
    h = polynomial_aggregation_(xc, k, mask)
    o = polynomial_selection_(xq, h)
    return o

def mask_mixer(h, mask):
    return (h * mask.unsqueeze(-1)).sum(dim=1, keepdims=True)/(1.e-7 + mask.unsqueeze(-1).sum(dim=1, keepdims=True))


def full_mask_mixer(h, mask):
    mask = mask.type(h.dtype)
    h = torch.einsum('bnd, bmn -> bmd', h, mask)  # b batch, n context tokens, m query tokens, d dim
    h = h / (1.e-7 + mask.sum(dim=2, keepdims=True))
    return h

@torch.compile
def gelu(x: torch.Tensor):
    return F.gelu(x)

@torch.compile
def po2(x: torch.Tensor):
    h1, h2 = gelu(x).chunk(2, dim=-1)
    h2 = h2 * h1
    return torch.cat([h1, h2], dim=-1)

@torch.compile
def po3(x: torch.Tensor):
    h1, h2, h3 = gelu(x).chunk(3, dim=-1)
    h2 = h2 * h1
    h3 = h3 * h2
    return torch.cat([h1, h2, h3], dim=-1)

@torch.compile
def po4(x: torch.Tensor):
    h1, h2, h3, h4 = gelu(x).chunk(4, dim=-1)
    h2 = h2 * h1
    h3 = h3 * h2
    h4 = h4 * h3
    return torch.cat([h1, h2, h3, h4], dim=-1)

def polynomial_aggregation_(x: torch.Tensor, k: int, mask=None):
    if k == 2:
        h = po2(x)
    elif k == 3:
        h = po3(x)
    elif k == 4:
        h = po4(x)
    else:
        h = list(gelu(x).chunk(k, dim=-1))
        for i in range(1, k):
            h[i] = h[i] * h[i-1]
        h = torch.cat(h, dim=-1)
    if mask is None:
        h = h.mean(dim=1, keepdims=True)
    else:
        if mask.dim() == 2:
            h = mask_mixer(h, mask.to(h.device))
        elif mask.dim() == 3:
            h = full_mask_mixer(h, mask.to(h.device))
        else:
            raise Exception('unsupported dim for mask (should be 2,3 or None)')
    return h

@torch.compile
def polynomial_selection_(x: torch.Tensor, h: torch.Tensor):
    return F.sigmoid(x) * h
