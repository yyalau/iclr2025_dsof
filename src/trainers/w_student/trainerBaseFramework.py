import torch
from trainers.trainerBaseERTD import TrainerBaseERTD

import warnings
warnings.filterwarnings('ignore')


class TrainerBaseFramework(TrainerBaseERTD):
    def __init__(self, args, main_model, student_model):
        super().__init__(args, main_model, student_model)

    def backward_lossT_train(self, prev,):
        lossT = self.criterion(prev['predT'], self.mainFFN.gt4update(prev))            
        lossT.backward()
        return lossT

    def backward_lossT_test(self, prev,):
        lossT = self.criterion(prev['predT'], self.mainFFN.gt4update(prev))            
        lossT.backward()
        return lossT
    
    def backward_lossFT(self, prev, curr):
        lossFT = self.criterion(prev['pred'], self.studentFFN.gt4update(curr))
        lossFT.backward()
        return lossFT
    
    def backward_tdlossFT(self, prev, curr):
        
        td_truth = []
        
        for i, k in enumerate(self.indices):
            # TODO: if FITS is student model, the following code cannot work.
            td_truth.append(
                torch.cat([ prev['true'][i:i+1,:k], 
                            self.mainFFN.td4update_pseudo(curr, k)], 
                            dim=1)
            )
        td_truth = torch.cat(td_truth, dim=0)
        
        lossFT = self.MSEReductNone(prev['pred'], td_truth) * self.discountedW
        lossFT = lossFT.mean()
        lossFT.backward()
        return lossFT
    

    def _select_optimizer(self):
        
        opts = super()._select_optimizer()
        
        opts['train'].append(self._get_optim(self.args.opt_student['train'], self.student_model.parameters()))
        opts['test']['batch'].append(self._get_optim(self.args.opt_student['test']['batch'], self.student_model.parameters()))
        opts['test']['online'].append(self._get_optim(self.args.opt_student['test']['online'], self.student_model.parameters()))

        return opts

    def test_step(self,dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark,):        

        if self.args.er:
            self.er(dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark)
        
        if self.args.td:
            self.td(dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark)

        with torch.no_grad():    
            curr = self._process_one_batch(dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark,  mode='test')
            curr.update({'pred': self.studentFFN.get_future(curr['pred'])})
        curr = self.clean(curr)            
        
        return curr

