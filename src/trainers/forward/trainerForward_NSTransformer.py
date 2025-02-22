# from einops import rearrange
import torch
from src.trainers.forward.trainerBaseForward import TrainerBaseForward
import warnings
warnings.filterwarnings('ignore')


class TrainerForward(TrainerBaseForward):
    def __init__(self, args, model, device,):
        super().__init__(args, model, device)

    def ffn(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark, mode='train'):
        
        f = lambda data: data.float().to(self.device)
        # import ipdb; ipdb.set_trace()
        batch_x, batch_x_mark, batch_y_mark = map(f, (batch_x, batch_x_mark, batch_y_mark))
        dec_inp = f(torch.cat([batch_y[:, :self.model.label_len, :],
                             torch.zeros_like(batch_y[:, -self.model.pred_len:, :])],
                            dim = 1))
        
        with torch.cuda.amp.autocast(enabled = self.args.use_amp):
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

        outputs.update({'true':  f(batch_y[:,-self.model.pred_len:,self.f_dim:]), })
        return outputs
