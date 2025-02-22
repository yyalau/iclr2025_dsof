

from src.models.wrappers import TS2VecEncoderWrapper
from src.models.encoders.OnlineEnc import TSEncoder
# from models.ts2vec.fsnet import TSEncoder
# from models.blocks.convblocks import 

from torch import nn
import torch

class Model(nn.Module):   
    def __init__(self, args, input_len):
        super().__init__()
        self.seq_len = input_len
        self.pred_len = args.pred_len
        self.encoder = TSEncoder(input_dims=args.enc_in+7,
                             output_dims=args.enc_output_dim,  # standard ts2vec backbone value
                             hidden_dims=args.enc_hidden_dim, # standard ts2vec backbone value
                             depth=args.depth,
                             args = args,)  

        self.dim = args.c_out * args.pred_len

        self.regressor = nn.Linear(args.enc_output_dim, self.dim)#.to(self.device)

        # import ipdb; ipdb.set_trace()

    def forward(self, x, embedding = False):
        # import ipdb; ipdb.set_trace()
        b, _, _ = x.shape
        z = x

        z = self.encoder(z)
        z = self.regressor(z)
        
        z = z.reshape(b, self.pred_len, -1)    

        return {'pred': z, }
    
    def store_grad(self):
        for name, layer in self.encoder.named_modules():    
            if 'PadConv' in type(layer).__name__:
                #print('{} - {}'.format(name, type(layer).__name__))
                layer.store_grad()
                
                
