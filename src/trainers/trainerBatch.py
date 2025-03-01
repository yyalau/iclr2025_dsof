from trainers.trainerBase import TrainerBase
import warnings
import torch
warnings.filterwarnings('ignore')


class Exp_TS2VecSupervised(TrainerBase):
    def __init__(self, args, main_model, student_model):
        
        assert student_model is None, 'Student model is not required for this framework'
        super().__init__(args, main_model)
    
    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark, mode='train'):

        outputs = self.mainFFN.ffn(dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark, mode)
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
        with torch.no_grad():    
            curr = self._process_one_batch(dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark,  mode='test')
                
        curr = self.clean(curr)            
        return curr
