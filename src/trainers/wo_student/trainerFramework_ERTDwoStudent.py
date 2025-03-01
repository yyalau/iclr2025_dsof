import torch
from trainers.trainerBaseERTD import TrainerBaseERTD
import warnings
warnings.filterwarnings('ignore')


class Exp_TS2VecSupervised(TrainerBaseERTD):
    def __init__(self, args, main_model, student_model):
        
        super().__init__(args, main_model, student_model)
        assert args.student_model['model'] is None, 'Student model exists. This framework is for training without student model.'

    def backward_lossT_train(self, prev):
        lossT = self.criterion(prev['predT'], self.mainFFN.gt4update(prev))            
        lossT.backward()
        return lossT
    
    def backward_lossT_test(self, prev):
        lossT = self.criterion(prev['predT'], self.mainFFN.gt4update(prev))            
        lossT.backward()
        return lossT
    
    def backward_lossFT(self, prev, curr):
        return
        # raise ModuleNotFoundError('This method should not be implemented in this framework.')

    def backward_tdlossFT(self, prev, curr):
        
        td_truth = []
        
        for i, k in enumerate(self.indices):
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
        
    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark, mode='train'):

        outputs = self.mainFFN.ffn(dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark, mode)
        outputs.update({'predT': outputs['pred'], })
        return outputs
    
    def train_step(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):
        outputs = self._process_one_batch(dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark, mode='train')
        
        loss = self.criterion(outputs['pred'], self.mainFFN.gt4update(outputs))
        loss.backward()
        
        for opt in self.opt['train']: opt.step()
        self.store_grad()
        self.update_ocp(outputs, outputs)
        
        return {'loss':loss.item()}


    def test_step(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark, model = None):        
        self.er(dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark)
        self.td(dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark)
        
        with torch.no_grad():    
            curr = self._process_one_batch(dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark,  mode='test')
                
        curr = self.clean(curr)            
        return curr