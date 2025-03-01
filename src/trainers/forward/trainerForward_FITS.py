# from einops import rearrange
import torch
from trainers.forward.trainerBaseForward import TrainerBaseForward
import warnings
warnings.filterwarnings('ignore')


class TrainerForward(TrainerBaseForward):
    def __init__(self, args, model, device,):
        super().__init__(args, model, device)

    def ffn(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark, mode='train'):


        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y[:,-self.args.pred_len:,self.f_dim:].float().to(self.device)

        with torch.cuda.amp.autocast(enabled = self.args.use_amp):
            outputs = self.model(batch_x)

        outputs.update({'true':  batch_y, 
                        'batch_xy': torch.cat([batch_x, batch_y], dim=1).float()})
        return outputs
    
    def gt4update(self, outputs):
        return outputs['batch_xy']
    
    def td4update_true(self, prev):
        cut = self.model.seq_len + 1  
        # returns sequence length + first ground truth       
        return prev['batch_xy'][:, :cut]
    
    def td4update_pseudo(self, curr, k):
        cut = self.model.seq_len
        return curr['predT'][:, cut:-k]
    
