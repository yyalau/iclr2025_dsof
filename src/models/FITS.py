import torch
import torch.nn as nn
import math

class TrendComponent(nn.Module):
    def __init__(self, configs) -> None:
        super().__init__()
        
        self.x_mean = None
        self.x_var = None
    
    def norm(self, x):
        self.x_mean = torch.mean(x, dim=1, keepdim=True)
        x = x - self.x_mean
        self.x_var=torch.var(x, dim=1, keepdim=True)+ 1e-5
        x = x / torch.sqrt(self.x_var) # [32,96,321]        
        return x
    
    def denorm(self, xy):
        return (xy) * torch.sqrt(self.x_var) +self.x_mean

    def forward(self, x, mode = 'norm'):
        
        if mode=='norm': return self.norm(x)
        if mode=='denorm': return self.denorm(x)


class LowFreqComponent(nn.Module):
    def __init__(self, configs) -> None:
        super().__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.remainder = (self.seq_len+self.pred_len)%2
        self.channels = configs.enc_in
        self.dominance_freq=configs.cut_freq # 720/24        
        self.length_ratio = (self.seq_len + self.pred_len + self.remainder )/self.seq_len
        
        
        self.freq_upsampler = nn.Linear(self.dominance_freq, 
                                        int(self.dominance_freq*self.length_ratio)).to(torch.cfloat) # complex layer for frequency upcampling]

    
    def forward(self, full_spectral):
        
        low_f_x = full_spectral[:, :self.dominance_freq]        
        low_f_xy_ = self.freq_upsampler(low_f_x.permute(0,2,1)).permute(0,2,1)
        # low_f_x.permute(0,2,1): [32,321,40] # low_f_xy_: [32,50,321]

        B, F, D = low_f_xy_.shape
        
        # zero padding
        low_f_xy = torch.zeros([B,math.ceil((self.seq_len+self.pred_len)/2+1), D],
                                 dtype=low_f_xy_.dtype).to(low_f_xy_.device)

        low_f_xy[:,:F,:]=low_f_xy_ 
        return low_f_xy


class Model(nn.Module):

    # FITS: Frequency Interpolation Time Series Forecasting

    def __init__(self, configs, input_len):
        super(Model, self).__init__()
        
        self.configs = configs
        self.seq_len = input_len
        self.pred_len = configs.pred_len
        self.individual = configs.individual
        self.channels = configs.enc_in
        
        assert configs.cut_freq > 0, 'cut_freq should be larger than 0'
    
        self.dominance_freq=configs.cut_freq # 720/24
        self.remainder = (self.seq_len+self.pred_len)%2
        self.length_ratio = (self.seq_len + self.pred_len + self.remainder )/self.seq_len
        self.trend_component = TrendComponent(configs)
        self.lf_component = LowFreqComponent(configs)
        
        

    def store_grad(self):
        return

    def forward(self, x):
        
        # TODO: use time information in time domain
        season = self.trend_component(x, 'norm')

        full_spectral = torch.fft.rfft(season, dim=1) # [32,49,321]        
        low_f_xy = self.lf_component(full_spectral)
        low_xy=torch.fft.irfft(low_f_xy, dim=1) * self.length_ratio # energy compemsation for the length change

        
        if self.remainder: low_xy=low_xy[:,:-1,:]       
        xy = self.trend_component(low_xy, 'denorm')               
        

        return {'pred': xy}