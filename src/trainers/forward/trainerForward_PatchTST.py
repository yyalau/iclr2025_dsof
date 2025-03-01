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
        x = batch_x

        with torch.cuda.amp.autocast(enabled = self.args.use_amp):
            # outputs = self.model(batch_x)
            if hasattr(self.args, 'ar_pred_len'):
                preds_temp = []
                for i in range(self.args.ar_pred_len):
                    outputs = self.model(x)                    
                    x = torch.cat(
                        [ 
                            x[:,1:,:], 
                            outputs['pred']
                        ], 
                            dim=1
                        )
                    preds_temp.append(outputs['pred'])
                outputs['pred'] = torch.cat(preds_temp, dim=1)
                batch_y = batch_y[:,-self.args.ar_pred_len:,self.f_dim:].float().to(self.device)
            else:
                outputs = self.model(x)
                batch_y = batch_y[:,-self.args.pred_len:,self.f_dim:].float().to(self.device)

        outputs.update({'true':  f(batch_y[:,-self.model.pred_len:,self.f_dim:]), })
        return outputs
    
