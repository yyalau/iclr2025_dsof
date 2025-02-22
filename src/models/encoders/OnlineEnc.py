from src.models.encoders.dilatedFSConvEnc import DilatedFSConvEnc
from torch import nn

class TSEncoder(nn.Module):
    def __init__(self, input_dims, output_dims, args, hidden_dims=64, depth=10, mask_mode='binomial', ):
                #  gamma=0.9, f_gamma = 0.3, tau=0.75, memory=True, memoryV2 = False, mem_size = 32,
                #  adapter_w = True, adapter_f = True):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.mask_mode = mask_mode
        self.input_fc = nn.Linear(input_dims, hidden_dims)
        self.feature_extractor = DilatedFSConvEnc(
            hidden_dims,
            [hidden_dims] * depth + [output_dims],
            kernel_size=args.kernel_size,         
            # args = args,
            )
        self.repr_dropout = nn.Dropout(p=0.1)

        # [64] * 10 + [320] = [64, 64, 64, 64, 64, 64, 64, 64, 64 ,64, 320] = 11 items
        # for i in range(len(...)) -> 0, 1, ..., 10
                
    def forward_time(self, x, mask=None):  # x: B x T x input_dims
        x = x.transpose(1, 2)
        nan_mask = ~x.isnan().any(axis=-1)
        x[~nan_mask] = 0
        
        # import ipdb; ipdb.set_trace()
        x = self.input_fc(x)  # B x T x Ch
        
        # generate & apply mask
        # mask = self.generate_mask(x, mask)
        
        # mask &= nan_mask
        # x[~mask] = 0
        
        # conv encoder
        x = x.transpose(1, 2)  # B x Ch x T
        x = self.repr_dropout(self.feature_extractor(x))  # B x Co x T
        x = x.transpose(1, 2)  # B x T x Co
        
        return x
    
    def forward(self, x, mask=None):  # x: B x T x input_dims
        
        nan_mask = ~x.isnan().any(axis=-1)
        x[~nan_mask] = 0
        x = self.input_fc(x)  # B x T x Ch

        # generate & apply mask
        # mask = self.generate_mask(x, mask)
        
        # mask &= nan_mask
        # x[~mask] = 0
        # conv encoder
        x = x.transpose(1, 2)  # B x Ch x T
        x = self.repr_dropout(self.feature_extractor(x))  # B x Co x T
        x = x.transpose(1, 2)  # B x T x Co
        
        return x[:, -1]