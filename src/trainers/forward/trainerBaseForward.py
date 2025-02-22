# from einops import rearrange
import torch
# from trainer.trainerBase import TrainerBase
import warnings
warnings.filterwarnings('ignore')


class TrainerBaseForward:
    def __init__(self, args, model, device,):
        
        self.args = args        
        self.model = model
        self.device = device    
        self.f_dim = -1 if self.args.features=='MS' else 0


    def ffn(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark, mode='train'):

        raise NotImplementedError
    
    def gt4update(self, outputs):
        return outputs['true']
    
    def td4update_true(self, prev, k):
        return prev['true'][:,:k]
    
    def td4update_pseudo(self, curr, k):
        return curr['predT'][:, :-k]
    
    def get_future(self, tensor):
        return tensor[:, -self.model.pred_len:]
    
    def store_grad(self):
        return

    def update_ocp(self, prev, curr):
        return 
    
    def extended_optimizer(self,):
        return {
            'train': [],
            'test':{
                'batch':[],
                'online':[],
            }
        }
