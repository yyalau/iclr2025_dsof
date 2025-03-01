import torch
from trainers.forward.trainerBaseForward import TrainerBaseForward
import warnings
warnings.filterwarnings('ignore')


class TrainerForward(TrainerBaseForward):
    def __init__(self, args, model, device,):
        super().__init__(args, model, device)

    def ffn(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark, mode='train'):

        f = lambda data: data.float().to(self.device)
        
        batch_x = f(batch_x)
        batch_x_mark = f(batch_x_mark)        

        with torch.cuda.amp.autocast(enabled = self.args.use_amp):
            outputs = self.model(batch_x, batch_x_mark)

        outputs.update({'true':  f(batch_y[:,-self.model.pred_len:,self.f_dim:]), })
        return outputs
    
