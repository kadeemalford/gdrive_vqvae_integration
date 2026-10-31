"""Two-pass index prior; non-anchor predictions cannot see non-anchor symbols."""
import torch
from torch import nn


def anchor_mask(height,width,device='cpu'):
    return (torch.arange(height,device=device)[:,None]+torch.arange(width,device=device)[None,:])%2==0


class CheckerboardPrior(nn.Module):
    def __init__(self,marginal):
        super().__init__()
        marginal=torch.as_tensor(marginal,dtype=torch.float32)
        if marginal.shape!=(2048,) or not torch.isfinite(marginal).all() or (marginal<=0).any():
            raise ValueError('Expected full-support marginal distribution')
        marginal=marginal/marginal.sum()
        self.register_buffer('marginal',marginal)
        self.embedding=nn.Embedding(2049,16,padding_idx=2048)
        self.context=nn.Sequential(nn.Conv2d(16,48,3,padding=1),nn.SiLU(),
                                   nn.Conv2d(48,48,3,padding=1),nn.SiLU())
        self.head=nn.Conv2d(48,2048,1)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)

    def forward(self,symbols):
        height,width=symbols.shape[-2:]
        anchors=anchor_mask(height,width,symbols.device)
        visible=torch.where(anchors,symbols,2048)
        features=self.embedding(visible).permute(0,3,1,2).contiguous()
        return self.head(self.context(features))+self.marginal.log()[None,:,None,None]

    @torch.no_grad()
    def conditional_probabilities(self,symbols):
        if next(self.parameters()).device.type!='cpu':
            raise ValueError('Reference entropy coding must use the shared CPU probability implementation')
        tensor=torch.from_numpy(symbols.astype('int64',copy=True))[None]
        logits=self(tensor)[0].permute(1,2,0)
        nonanchors=~anchor_mask(*symbols.shape)
        return logits[nonanchors].softmax(dim=-1).contiguous().numpy()
