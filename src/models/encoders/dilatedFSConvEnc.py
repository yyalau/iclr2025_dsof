from torch import nn
from src.models.blocks.FSConv import FSConv
import torch.nn.functional as F

    
class FSConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, final=False, gamma=0.9):
        super().__init__()
        self.conv1 = FSConv(in_channels, out_channels, kernel_size, dilation=dilation, gamma=gamma)
        self.conv2 = FSConv(out_channels, out_channels, kernel_size, dilation=dilation, gamma=gamma)
        self.projector = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels or final else None
    
    def ctrl_params(self):  
        c_iter = chain(self.conv1.controller.parameters(), self.conv1.calib_w.parameters(), 
                self.conv1.calib_b.parameters(), self.conv1.calib_f.parameters(),
                self.conv2.controller.parameters(), self.conv2.calib_w.parameters(), 
                self.conv2.calib_b.parameters(), self.conv2.calib_f.parameters())

        return c_iter 
       

    def forward(self, x):
        residual = x if self.projector is None else self.projector(x)
        x = F.gelu(x)
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        return x + residual


class DilatedFSConvEnc(nn.Module):
    def __init__(self, in_channels, channels, kernel_size, gamma=0.9):
        super().__init__()
        self.net = nn.Sequential(*[
            FSConvBlock(
                channels[i-1] if i > 0 else in_channels,
                channels[i],
                kernel_size=kernel_size,
                dilation=2**i,
                final=(i == len(channels)-1), gamma=gamma
            )
            for i in range(len(channels))
        ])
    def ctrl_params(self):
        ctrl = []
        for l in self.net:
            ctrl.append(l.ctrl_params())
        c = chain(*ctrl)
        for p in c:
            yield p
    def forward(self, x):
        return self.net(x)

        
