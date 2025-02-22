from torch import nn

class TS2VecEncoderWrapper(nn.Module):
    def __init__(self, encoder, mask):
        super().__init__()
        self.encoder = encoder
        self.mask = mask

    def forward(self, input):
        
        return self.encoder(input, mask=self.mask)[:, -1]
    