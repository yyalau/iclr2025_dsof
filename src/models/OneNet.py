import torch
from torch import nn
from src.models.wrappers import TS2VecEncoderWrapper
from src.models.encoders.OnlineEnc import TSEncoder

# from models.ts2vec.fsnet import TSEncoder, GlobalLocalMultiscaleTSEncoder

class Model(nn.Module):
    def __init__(self, args, input_len):
        super().__init__()
        # self.device = device
        self.seq_len = input_len
        self.pred_len = args.pred_len
        self.encoder_time = TSEncoder(input_dims=args.seq_len,
                             output_dims=args.enc_output_dim,  # standard ts2vec backbone value
                             hidden_dims=args.enc_hidden_dim, # standard ts2vec backbone value
                             depth=args.depth,
                             args = args) 
        # self.encoder_time = TS2VecEncoderWrapper(encoder, mask='all_true')#.to(self.device)
        self.regressor_time = nn.Linear(args.enc_output_dim, args.pred_len) #.to(self.device)
        
        self.encoder = TSEncoder(input_dims=args.enc_in + 7,
                             output_dims=args.enc_output_dim,  # standard ts2vec backbone value
                             hidden_dims=args.enc_hidden_dim, # standard ts2vec backbone value
                             depth=args.depth,
                             args = args) 
        
        # import ipdb; ipdb.set_trace()

        # self.encoder = TS2VecEncoderWrapper(encoder, mask='all_true')#.to(self.device)
        
        self.dim = args.c_out * args.pred_len
        
        self.regressor = nn.Linear(args.enc_output_dim, self.dim)#.to(self.device)
    
    
    def forward_weight(self, x, x_mark):
        
        b, _, _ = x.shape
        
        
        rep = self.encoder_time.forward_time(x)
        y1 = self.regressor_time(rep).transpose(1, 2)
        # y1 = rearrange(y, 'b t d -> b (t d)')
        
        x = torch.cat([x, x_mark], dim=-1)
        rep2 = self.encoder(x)#[:, -1]
        y2 = self.regressor(rep2).reshape(b, self.pred_len, -1)
        
        # import ipdb; ipdb.set_trace()
    
        return { "y1": y1, "y2": y2}
        
    def store_grad(self):
        for name, layer in self.encoder.named_modules():    
            if 'PadConv' in type(layer).__name__:
                #print('{} - {}'.format(name, type(layer).__name__))
                layer.store_grad()
        for name, layer in self.encoder_time.named_modules():    
            if 'PadConv' in type(layer).__name__:
                #print('{} - {}'.format(name, type(layer).__name__))
                layer.store_grad()
