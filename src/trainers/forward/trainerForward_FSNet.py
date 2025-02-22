# from einops import rearrange
import torch
from src.trainers.forward.trainerBaseForward import TrainerBaseForward
import warnings
warnings.filterwarnings('ignore')


class TrainerForward(TrainerBaseForward):
    def __init__(self, args, model, device,):
        super().__init__(args, model, device)
    
    def store_grad(self,):
        self.model.store_grad()

    def ffn(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark, mode='train'):

        batch_x = batch_x.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        x = torch.cat([batch_x, batch_x_mark], dim=-1)
        batch_y = batch_y[:,-self.args.pred_len:,self.f_dim:].float().to(self.device)

        with torch.cuda.amp.autocast(enabled = self.args.use_amp):
            outputs = self.model(x)

        outputs.update({'true':  batch_y, })
        return outputs
    
